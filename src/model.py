import torch
import torch.nn as nn
import timm
from timm.data import resolve_model_data_config, create_transform
from transformers import (
    AutoModelForCausalLM, 
    PreTrainedModel, 
    PretrainedConfig, 
    AutoTokenizer
)
from typing import Optional, List, Union
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Configuration Class
# -----------------------------------------------------------------------------
class NanoQwenVLConfig(PretrainedConfig):
    model_type = "nano_qwen_vl"
    def __init__(
        self,
        llm_model_id="google/gemma-3-270m-it",
        vision_model_id="naflexvit_base_patch16_siglip.v2_webli",
        freeze_vision=True,
        freeze_llm=False,
        use_flash_attention=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.llm_model_id = llm_model_id
        self.vision_model_id = vision_model_id
        self.freeze_vision = freeze_vision
        self.freeze_llm = freeze_llm
        self.use_flash_attention = use_flash_attention

# -----------------------------------------------------------------------------
# 2. Processor (Using TIMM for PE-Core)
# -----------------------------------------------------------------------------
class NanoQwenProcessor:
    def __init__(self, vision_model_id, llm_model_id):
        # PE-Core uses TIMM transforms
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
        
        # Get transform from TIMM
        data_config = resolve_model_data_config(timm.create_model(vision_model_id, pretrained=False))
        self.image_transform = create_transform(**data_config, is_training=False)

    def __call__(self, text: Union[str, List[str]], images: Union[Image.Image, List[Image.Image]] = None):
        if isinstance(text, str):
            text = [text]
        if images and isinstance(images, Image.Image):
            images = [images]

        # 1. Process Text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        
        # 2. Process Images
        if images:
            # Apply TIMM transform and stack
            pixel_values = torch.stack([self.image_transform(img) for img in images])
            inputs['pixel_values'] = pixel_values

        return inputs

# -----------------------------------------------------------------------------
# 3. Projector Module (DeepSpeed compatible)
# -----------------------------------------------------------------------------
class VisionProjector(nn.Module):
    """Projects vision features to LLM dimension. Separate class for DeepSpeed compatibility."""
    def __init__(self, vision_dim, llm_dim):
        super().__init__()
        self.norm = nn.LayerNorm(vision_dim)
        self.fc1 = nn.Linear(vision_dim, llm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# -----------------------------------------------------------------------------
# 4. Model Definition
# -----------------------------------------------------------------------------
class NanoQwenVL(PreTrainedModel):
    config_class = NanoQwenVLConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Load LLM
        print(f"Loading LLM: {config.llm_model_id}...")
        llm_kwargs = {
            "trust_remote_code": True,
        }
        if config.use_flash_attention:
            llm_kwargs["attn_implementation"] = "flash_attention_2"
            print("  -> Using Flash Attention 2")
        
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_id, 
            **llm_kwargs
        )
        
        # Load NaFlexViT from TIMM
        print(f"Loading Vision Model: {config.vision_model_id}...")
        self.vision_tower = timm.create_model(
            config.vision_model_id,
            pretrained=True,
            use_naflex=True,  # Enable NaFlex for flexible resolution
            dynamic_img_size=True,  # Allow dynamic image sizes
            dynamic_img_pad=True,   # Enable dynamic padding
            num_classes=0,  # Remove classification head
            # Note: Don't set global_pool='' as it causes weight loading issues with SigLIP models
            # Instead, we'll use forward_features() to get patch embeddings
        )

        if config.freeze_vision:
            self.vision_tower.requires_grad_(False)
        if config.freeze_llm:
            self.llm.requires_grad_(False)

        # Enable gradient checkpointing for memory efficiency
        self.llm.gradient_checkpointing_enable()
        self.vision_tower.set_grad_checkpointing(enable=True)

        # Dimensions
        # NaFlexViT Base outputs 768-dim embeddings
        self.vision_dim = 768
        self.llm_dim = self.llm.config.hidden_size  # Gemma 270M: 1024
        
        # Projector (using proper Module class for DeepSpeed compatibility)
        self.projector = VisionProjector(self.vision_dim, self.llm_dim)

    def get_input_embeddings(self):
        """Required by PreTrainedModel/PEFT - delegate to LLM."""
        return self.llm.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """Required by PreTrainedModel/PEFT - delegate to LLM."""
        self.llm.set_input_embeddings(value)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        labels=None,
        image_token_id=None,  # Token ID for <start_of_image>
        **kwargs
    ):
        """
        Forward pass with Gemma 3 style image token replacement.
        Image tokens in input_ids are replaced with projected vision embeddings.
        """
        from src.data import NUM_IMAGE_TOKENS
        
        # Get text embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if pixel_values is not None and image_token_id is not None:
            # 1. Get vision features
            vision_outputs = self.vision_tower.forward_features(pixel_values)
            
            # Ensure 3D: [B, num_patches, vision_dim]
            if vision_outputs.dim() == 2:
                vision_outputs = vision_outputs.unsqueeze(0)
            
            # 2. Project to LLM dimension and pool to 256 tokens per image
            # vision_outputs: [B, num_patches, vision_dim] -> [B, 256, llm_dim]
            batch_size = vision_outputs.shape[0]
            num_patches = vision_outputs.shape[1]
            
            # Adaptive pooling to get exactly 256 tokens
            # Reshape: [B, num_patches, dim] -> [B, dim, num_patches]
            vision_outputs = vision_outputs.transpose(1, 2)
            # Pool: [B, dim, num_patches] -> [B, dim, 256]
            vision_pooled = nn.functional.adaptive_avg_pool1d(vision_outputs, NUM_IMAGE_TOKENS)
            # Reshape back: [B, dim, 256] -> [B, 256, dim]
            vision_pooled = vision_pooled.transpose(1, 2)
            
            # Project to LLM dimension
            image_embeds = self.projector(vision_pooled)  # [B, 256, llm_dim]
            
            # 3. Replace image tokens in inputs_embeds
            for batch_idx in range(inputs_embeds.shape[0]):
                # Find positions of image tokens
                image_mask = (input_ids[batch_idx] == image_token_id)
                image_positions = torch.where(image_mask)[0]
                
                if len(image_positions) > 0:
                    # Get image embedding for this batch item
                    img_emb = image_embeds[batch_idx if batch_idx < batch_size else 0]
                    
                    # Replace embeddings at image token positions
                    num_to_replace = min(len(image_positions), img_emb.shape[0])
                    for j in range(num_to_replace):
                        pos = image_positions[j]
                        inputs_embeds[batch_idx, pos] = img_emb[j]
                    
                    # Set labels to -100 for image positions (don't compute loss)
                    if labels is not None:
                        labels[batch_idx, image_positions[:num_to_replace]] = -100
        
        # Clean kwargs
        kwargs.pop('inputs_embeds', None)
        kwargs.pop('attention_mask', None) 
        kwargs.pop('labels', None)
        kwargs.pop('input_ids', None)
        kwargs.pop('return_dict', None)
        kwargs.pop('pixel_values', None)
        kwargs.pop('image_token_id', None)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return self.llm.prepare_inputs_for_generation(input_ids, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        image_token_id=None,  # Token ID for <start_of_image>
        **kwargs
    ):
        """Generate with Gemma 3 style image token replacement."""
        from src.data import NUM_IMAGE_TOKENS
        
        # Get text embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if pixel_values is not None and image_token_id is not None:
            # Get vision features
            vision_outputs = self.vision_tower.forward_features(pixel_values)
            
            if vision_outputs.dim() == 2:
                vision_outputs = vision_outputs.unsqueeze(0)
            
            # Pool to 256 tokens
            batch_size = vision_outputs.shape[0]
            vision_outputs = vision_outputs.transpose(1, 2)
            vision_pooled = nn.functional.adaptive_avg_pool1d(vision_outputs, NUM_IMAGE_TOKENS)
            vision_pooled = vision_pooled.transpose(1, 2)
            
            image_embeds = self.projector(vision_pooled)
            
            # Replace image tokens
            for batch_idx in range(inputs_embeds.shape[0]):
                image_mask = (input_ids[batch_idx] == image_token_id)
                image_positions = torch.where(image_mask)[0]
                
                if len(image_positions) > 0:
                    img_emb = image_embeds[batch_idx if batch_idx < batch_size else 0]
                    num_to_replace = min(len(image_positions), img_emb.shape[0])
                    for j in range(num_to_replace):
                        pos = image_positions[j]
                        inputs_embeds[batch_idx, pos] = img_emb[j]

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
