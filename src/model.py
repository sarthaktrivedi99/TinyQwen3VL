import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PretrainedConfig
import open_clip

class NanoQwenVLConfig(PretrainedConfig):
    model_type = "nano_qwen_vl"
    def __init__(
        self,
        llm_model_id="Qwen/Qwen3-0.6B",
        vision_model_id="hf-hub:timm/PE-Core-S-16-384",
        freeze_vision=True,
        freeze_llm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.llm_model_id = llm_model_id
        self.vision_model_id = vision_model_id
        self.freeze_vision = freeze_vision
        self.freeze_llm = freeze_llm

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
        
        # Load Vision Model
        print(f"Loading Vision Model: {config.vision_model_id}...")
        # OpenCLIP loading
        # User requested: "keep the model name same just with hf-hub: added to the name"
        # We assume this means creating the model with that name.
        self.vision_tower = open_clip.create_model(
            config.vision_model_id,
            pretrained=None # 'frozen' removed
        )
        # Check if we need to load specific pretrained checkpoint if not auto-loaded
        
        # Freeze components if requested
        if config.freeze_vision:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
                
        if config.freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False

        # Projection Layer
        # We need to map vision output dim to LLM embedding dim
        # OpenCLIP vision tower usually has .visual or is the visual model itself if create_model returned just visual?
        # Standard open_clip.create_model returns the full CLIP model (text + visual).
        # We only want visual.
        
        if hasattr(self.vision_tower, 'visual'):
            self.vision_module = self.vision_tower.visual
        else:
            self.vision_module = self.vision_tower

        # Check visual dim
        # open_clip models often have output_dim (projection) or transformer width.
        # PE-Core is a transformer. 
        # width might be hidden size. output_dim might be projection dim.
        # We usually want the storage/transformer dim before projection for VLMs, or the projected one?
        # Let's use the output of the visual tower.
        
        # Let's try to infer dim by running a dummy input or checking config
        # open_clip / timm wrappers might have different attributes
        if hasattr(self.vision_module, 'output_dim'):
            self.vision_dim = self.vision_module.output_dim
        elif hasattr(self.vision_module, 'num_features'):
            self.vision_dim = self.vision_module.num_features
        elif hasattr(self.vision_module, 'embed_dim'):
            self.vision_dim = self.vision_module.embed_dim
        elif hasattr(self.vision_module, 'width'):
            self.vision_dim = self.vision_module.width
        else:
             # Fallback
             print("Warning: Could not detect vision dimension. Defaulting to 512 (based on error).")
             self.vision_dim = 512 
        
        # If the vision tower projects to common space, use that.
        # But for VLM often we want the patch embeddings (features).
        
        self.llm_dim = self.llm.get_input_embeddings().weight.shape[1]
        
        print(f"Vision Dim: {self.vision_dim}, LLM Dim: {self.llm_dim}")

        self.projector = nn.Sequential(
            nn.Linear(self.vision_dim, self.llm_dim),
            nn.GELU(),
            nn.Linear(self.llm_dim, self.llm_dim)
        )

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def forward(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        labels=None,
        **kwargs
    ):
        """
        input_ids: [B, SeqLen]
        pixel_values: [B, C, H, W]
        """
        
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        if pixel_values is not None:
            # Encode images
            # open_clip visual.forward usually returns projected global feature [B, Dim].
            # Getting patch tokens might require different method or inspecting code.
            # Many standard CLIP models don't return patch tokens in default forward().
            # However, simpler to use what it returns (pooled) or hack it?
            # For "Nano" VLM, pooled features (1 token per image) is simplest, but maybe fewer details.
            # PE-Core is "Perception Encoder", maybe it behaves differently. 
            # Let's assume global feature for now [B, VisionDim].
            
            vision_features = self.vision_module(pixel_values)
            
            # Check shape
            if vision_features.ndim == 2: #[B, Dim]
                 vision_features = vision_features.unsqueeze(1) # [B, 1, Dim]
            
            # Project
            image_embeds = self.projector(vision_features) # [B, 1, LLM_Dim]
            
            # We assume input_ids has special tokens where images should be merged.
            # SIMPLE APPROACH for now: Prepend image embeddings to text embeddings.
            # In a real rigorous setup we'd replace specific tokens.
            # Assuming 'pixel_values' provided implies we prepend.
            
            inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)
            
            if attention_mask is not None:
                # Extend attention mask for image tokens
                # 1s for image tokens
                image_mask = torch.ones(
                    image_embeds.shape[:2], 
                    dtype=attention_mask.dtype, 
                    device=attention_mask.device
                )
                attention_mask = torch.cat([image_mask, attention_mask], dim=1)
            
            if labels is not None:
                # Ignore loss for image tokens
                image_labels = torch.full(
                    image_embeds.shape[:2], 
                    -100, 
                    dtype=labels.dtype, 
                    device=labels.device
                )
                labels = torch.cat([image_labels, labels], dim=1)
                
        # Clean kwargs to avoid multiple value errors when calling llm
        # defined args in forward() don't end up in kwargs, but extra args from Trainer do.
        # Trainer might pass return_dict, output_attentions, etc.
        
        # We explicitly pass: inputs_embeds, attention_mask, labels, return_dict.
        # So we must remove them from kwargs if present (except return_dict which we want to force or default).
        
        kwargs.pop('inputs_embeds', None)
        kwargs.pop('attention_mask', None) 
        kwargs.pop('labels', None)
        # We explicitly pass return_dict=True, so remove it from kwargs if present
        kwargs.pop('return_dict', None)
        
        # Also remove input_ids if it somehow got in (it shouldn't if it's an arg)
        kwargs.pop('input_ids', None)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
            **kwargs
        )
        
        return outputs

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.llm.prepare_inputs_for_generation(*args, **kwargs)

    _supports_gradient_checkpointing = True

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, PreTrainedModel):
            module.gradient_checkpointing = value
        # Also set for LLM specifically if it's not the module passed (which is self)
        # When calling gradient_checkpointing_enable on self, it calls this with module=self usually?
        # Actually transformers iterates modules or expects this method to handle logic.
        # But standard PreTrainedModel implementation uses apply on modules if generic?
        # Let's just delegate explicitely.
        if hasattr(self.llm, "gradient_checkpointing_enable"):
             # If value is True/False, we can't directly use enable/disable easily if we want to pass kwargs.
             # But usually value is boolean.
             self.llm.gradient_checkpointing = value
             # We might need to recursively set it if we rely on underlying model's logic.
             if value:
                 self.llm.gradient_checkpointing_enable()
             else:
                 self.llm.gradient_checkpointing_disable()
        
        # Vision tower
        if hasattr(self.vision_tower, "set_grad_checkpointing"):
            self.vision_tower.set_grad_checkpointing(value)
        elif hasattr(self.vision_tower, "gradient_checkpointing"):
            self.vision_tower.gradient_checkpointing = value

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        **kwargs
    ):
        # 1. Get Text Embeddings
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        
        # 2. Get Image Embeddings if pixel_values provided
        if pixel_values is not None:
             vision_features = self.vision_module(pixel_values)
             if vision_features.ndim == 2: #[B, Dim]
                 vision_features = vision_features.unsqueeze(1) # [B, 1, Dim]
             
             image_embeds = self.projector(vision_features) # [B, 1, LLM_Dim]
             
             # Concatenate: [Image, Text]
             # This aligns with our forward pass logic where we assumed prepending
             inputs_embeds = torch.cat([image_embeds, inputs_embeds], dim=1)
             
             # Update attention_mask
             if attention_mask is not None:
                image_mask = torch.ones(
                    image_embeds.shape[:2], 
                    dtype=attention_mask.dtype, 
                    device=attention_mask.device
                )
                attention_mask = torch.cat([image_mask, attention_mask], dim=1)
        
        # 3. Generate
        # We pass inputs_embeds to the underlying LLM's generate
        # We must NOT pass input_ids if we pass inputs_embeds, usually.
        # But we might need 'input_ids' for some generation config defaults? 
        # Usually 'inputs_embeds' is sufficient.
        
        # Note: self.llm.generate might expect 'input_ids' for some checks.
        # Use inputs_embeds.
        
        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs
        )
        
        return outputs
