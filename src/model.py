import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoModel,
    PreTrainedModel, 
    PretrainedConfig, 
    AutoTokenizer,
    AutoImageProcessor
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
        llm_model_id="Qwen/Qwen3-0.6b",
        vision_model_id="moonshotai/MoonViT-SO-400M", # <--- Switched to MoonViT
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
# 2. Processor (Simplifed for MoonViT)
# -----------------------------------------------------------------------------
class NanoQwenProcessor:
    def __init__(self, vision_model_id, llm_model_id):
        # MoonViT has a custom image processor that handles native resolution logic
        self.image_processor = AutoImageProcessor.from_pretrained(
            vision_model_id, 
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)

    def __call__(self, text: Union[str, List[str]], images: Union[Image.Image, List[Image.Image]] = None):
        if isinstance(text, str):
            text = [text]
        if images and isinstance(images, Image.Image):
            images = [images]

        # 1. Process Text
        inputs = self.tokenizer(text, return_tensors="pt", padding=True)
        
        # 2. Process Images (Native Resolution)
        if images:
            # MoonViT processor returns 'pixel_values' and 'image_grid_hws'
            image_inputs = self.image_processor(images, return_tensors="pt")
            
            inputs['pixel_values'] = image_inputs['pixel_values']
            inputs['image_grid_hws'] = image_inputs['image_grid_hws'] # Vital for MoonViT

        return inputs

# -----------------------------------------------------------------------------
# 3. Model Definition
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
        
        # Load MoonViT (Needs trust_remote_code=True)
        print(f"Loading Vision Model: {config.vision_model_id}...")
        self.vision_tower = AutoModel.from_pretrained(
            config.vision_model_id,
            trust_remote_code=True
        )

        if config.freeze_vision:
            self.vision_tower.requires_grad_(False)
        if config.freeze_llm:
            self.llm.requires_grad_(False)

        # Dimensions
        # MoonViT-SO-400M output dim is 1152 (same as SigLIP)
        self.vision_dim = 1152 
        self.llm_dim = self.llm.config.hidden_size
        
        # Projector
        self.projector = nn.Sequential(
            nn.Linear(self.vision_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim)
        )

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        image_grid_hws=None, # <--- New argument required by MoonViT
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if pixel_values is not None:
            # 1. Forward Vision Tower
            # MoonViT forward signature: (pixel_values, image_grid_hws)
            # It returns a list of tensors (one per image) because lengths vary
            # Each tensor has shape [num_tokens, num_sub_patches, vision_dim] e.g., [1092, 4, 1152]
            vision_outputs = self.vision_tower(pixel_values, image_grid_hws)
            
            image_embeds_list = []
            
            # Process each image feature independently
            for i, vis_feat in enumerate(vision_outputs):
                # vis_feat shape: [Num_Tokens, 4, 1152]
                # Flatten the sub-patch dimension: [Num_Tokens * 4, 1152]
                if vis_feat.dim() == 3:
                    num_tokens, num_sub, dim = vis_feat.shape
                    vis_feat = vis_feat.reshape(num_tokens * num_sub, dim)
                
                # Project to LLM dimension
                proj_feat = self.projector(vis_feat)  # [Num_Tokens * 4, LLM_Dim]
                image_embeds_list.append(proj_feat)

            # 2. Merge into LLM sequence
            # Simplified Strategy: Prepend to text.
            # Since batch items have different image token lengths, we have to construct
            # the inputs_embeds batch carefully or use left-padding.
            
            # Simple approach: Create a new inputs_embeds tensor
            # We assume Batch Size is 1 for "Nano" demo simplicity, 
            # or we handle padding manually.
            
            new_inputs_embeds = []
            new_attention_mask = []
            new_labels = []

            for i in range(len(inputs_embeds)):
                # Get components for this batch item
                txt_emb = inputs_embeds[i] # [Seq, Dim]
                img_emb = image_embeds_list[i] # [Img_Seq, Dim]
                
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
            # (Use torch.nn.utils.rnn.pad_sequence or manual padding)
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
        
        # Explicitly remove vision args so LLM doesn't complain
        kwargs.pop('pixel_values', None)
        kwargs.pop('image_grid_hws', None)

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
        image_grid_hws=None,
        attention_mask=None,
        **kwargs
    ):
        # Re-implement simple logic for generation
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if pixel_values is not None:
            vision_outputs = self.vision_tower(pixel_values, image_grid_hws)
            
            new_inputs_embeds = []
            new_attention_mask = []
            
            for i in range(len(inputs_embeds)):
                vis_feat = vision_outputs[i]
                # Flatten 3D output: [Num_Tokens, 4, 1152] -> [Num_Tokens * 4, 1152]
                if vis_feat.dim() == 3:
                    num_tokens, num_sub, dim = vis_feat.shape
                    vis_feat = vis_feat.reshape(num_tokens * num_sub, dim)
                
                img_emb = self.projector(vis_feat)
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
