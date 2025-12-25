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
        **kwargs
    ):
        super().__init__(**kwargs)
        self.llm_model_id = llm_model_id
        self.vision_model_id = vision_model_id
        self.freeze_vision = freeze_vision
        self.freeze_llm = freeze_llm

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
        self.llm = AutoModelForCausalLM.from_pretrained(
            config.llm_model_id, 
            trust_remote_code=True
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
        **kwargs
    ):
        
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if pixel_values is not None:
            # 1. Forward Vision Tower using forward_features to get patch embeddings
            # NaFlexViT outputs: [batch, num_tokens, vision_dim] e.g., [B, num_patches, 768]
            vision_outputs = self.vision_tower.forward_features(pixel_values)
            
            # Ensure 3D tensor [B, num_patches, dim] - some models may return 2D
            if vision_outputs.dim() == 2:
                vision_outputs = vision_outputs.unsqueeze(0)  # Add batch dim if missing
            
            # 2. Project to LLM dimension [batch, num_tokens, llm_dim]
            image_embeds = self.projector(vision_outputs)  # [B, num_patches, LLM_Dim]
            
            # 3. Merge into LLM sequence
            # Prepend image embeddings to text for each item in batch
            new_inputs_embeds = []
            new_attention_mask = []
            new_labels = []

            for i in range(len(inputs_embeds)):
                # Get components for this batch item
                txt_emb = inputs_embeds[i]  # [Seq, Dim]
                img_emb = image_embeds[i]    # [num_patches, Dim]
                
                # Ensure 2D tensors for concatenation
                if img_emb.dim() == 1:
                    img_emb = img_emb.unsqueeze(0)  # [1, Dim] - single token
                
                # Concatenate [Image, Text]
                combined_emb = torch.cat([img_emb, txt_emb], dim=0)
                new_inputs_embeds.append(combined_emb)
                
                # Handle Attention Mask
                if attention_mask is not None:
                    cur_mask = attention_mask[i]
                    # Create mask for image (ones)
                    img_mask = torch.ones(img_emb.shape[0], device=cur_mask.device, dtype=cur_mask.dtype)
                    combined_mask = torch.cat([img_mask, cur_mask], dim=0)
                    new_attention_mask.append(combined_mask)
                
                # Handle Labels
                if labels is not None:
                    cur_lbl = labels[i]
                    img_lbl = torch.full((img_emb.shape[0],), -100, device=cur_lbl.device, dtype=cur_lbl.dtype)
                    combined_lbl = torch.cat([img_lbl, cur_lbl], dim=0)
                    new_labels.append(combined_lbl)

            # Pad the batch to the longest sequence
            from torch.nn.utils.rnn import pad_sequence
            
            inputs_embeds = pad_sequence(new_inputs_embeds, batch_first=True)
            if attention_mask is not None:
                # Pad with 0
                attention_mask = pad_sequence(new_attention_mask, batch_first=True, padding_value=0)
            if labels is not None:
                # Pad with -100
                labels = pad_sequence(new_labels, batch_first=True, padding_value=-100)

        # Clean kwargs
        kwargs.pop('inputs_embeds', None)
        kwargs.pop('attention_mask', None) 
        kwargs.pop('labels', None)
        kwargs.pop('input_ids', None)
        kwargs.pop('return_dict', None)
        
        # Explicitly remove vision args so LLM doesn't complain
        kwargs.pop('pixel_values', None)

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
        **kwargs
    ):
        # Re-implement simple logic for generation
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if pixel_values is not None:
            # PE-Core forward: [batch, num_tokens, vision_dim]
            vision_outputs = self.vision_tower(pixel_values)
            image_embeds = self.projector(vision_outputs)  # [B, 576, LLM_Dim]
            
            new_inputs_embeds = []
            new_attention_mask = []
            
            for i in range(len(inputs_embeds)):
                img_emb = image_embeds[i]  # [576, Dim]
                txt_emb = inputs_embeds[i]
                
                combined_emb = torch.cat([img_emb, txt_emb], dim=0)
                new_inputs_embeds.append(combined_emb)
                
                if attention_mask is not None:
                    cur_mask = attention_mask[i]
                    img_mask = torch.ones(img_emb.shape[0], device=cur_mask.device, dtype=cur_mask.dtype)
                    new_attention_mask.append(torch.cat([img_mask, cur_mask], dim=0))

            from torch.nn.utils.rnn import pad_sequence
            inputs_embeds = pad_sequence(new_inputs_embeds, batch_first=True)
            if attention_mask is not None:
                attention_mask = pad_sequence(new_attention_mask, batch_first=True, padding_value=0)

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
