import torch
import torch.nn as nn
import timm
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PretrainedConfig,
)


class TinyQwen3VLConfig(PretrainedConfig):
    model_type = "tiny_qwen3_vl"

    def __init__(
        self,
        llm_model_id="Qwen/Qwen3-0.6B",
        vision_model_id="timm/naflexvit_base_patch16_siglip.v2_webli",
        freeze_vision=True,
        freeze_llm=False,
        vision_hidden_size=768,
        patch_size=16,
        spatial_merge_size=2,
        image_token_id=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.llm_model_id = llm_model_id
        self.vision_model_id = vision_model_id
        self.freeze_vision = freeze_vision
        self.freeze_llm = freeze_llm
        self.vision_hidden_size = vision_hidden_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.image_token_id = image_token_id


class PatchMerger(nn.Module):
    """
    Qwen3VL-style PatchMerger: 2x2 spatial merge + 2-layer MLP.

    Matches the reference implementation exactly:
      LayerNorm(input_dim) → Linear(merged_dim, merged_dim) → GELU → Linear(merged_dim, output_dim)
    NO output LayerNorm — the output feeds directly into the LLM.
    """

    def __init__(self, vision_dim, llm_dim, merge_size=2):
        super().__init__()
        self.merge_size = merge_size
        merged_dim = vision_dim * (merge_size ** 2)
        self.norm = nn.LayerNorm(vision_dim)
        self.fc1 = nn.Linear(merged_dim, merged_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(merged_dim, llm_dim)

    def forward(self, x, grid_h, grid_w):
        B, N, D = x.shape
        ms = self.merge_size

        # Pre-merge LayerNorm (applied before spatial grouping, matching reference)
        x = self.norm(x)

        # Spatial 2x2 grouping
        x = x.reshape(B, grid_h, grid_w, D)
        x = x.reshape(B, grid_h // ms, ms, grid_w // ms, ms, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, (grid_h // ms) * (grid_w // ms), ms * ms * D)

        # MLP projection (no output norm — matches reference)
        return self.fc2(self.act(self.fc1(x)))


class TinyQwen3VL(PreTrainedModel):
    """
    Tiny Qwen3VL = SigLIP-2 NaFlexViT + PatchMerger projector + Qwen3 LLM.

    Architecture follows the reference Qwen3VL implementation:
      1. ViT extracts patch features
      2. PatchMerger does 2×2 spatial merge + MLP projection to LLM dim
      3. Vision features replace <|image_pad|> in text embeddings
      4. Combined embeddings go through the LLM (standard HF forward)

    Key: pixel_values are cast to LLM dtype before vision forward,
    matching the reference which does `pixels.to(input_embeds.dtype)`.
    """
    config_class = TinyQwen3VLConfig

    def __init__(self, config):
        super().__init__(config)

        # Vision encoder (frozen by default)
        self.vision_tower = timm.create_model(
            config.vision_model_id, pretrained=True,
            dynamic_img_size=True, num_classes=0,
        )

        # LLM
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_model_id)
        llm_dim = self.llm.config.hidden_size

        # PatchMerger projector (matches reference: no output LayerNorm)
        self.projector = PatchMerger(
            config.vision_hidden_size, llm_dim, config.spatial_merge_size,
        )

        # Freeze
        if config.freeze_vision:
            self.vision_tower.requires_grad_(False)
        if config.freeze_llm:
            self.llm.requires_grad_(False)

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    def _get_vision_features(self, pixel_values):
        """Run ViT, return [B, N, D] features + grid dims."""
        B, C, H, W = pixel_values.shape
        ps = self.config.patch_size
        return self.vision_tower.forward_features(pixel_values), H // ps, W // ps

    def _project_and_flatten(self, feats, gh, gw, patch_mask=None):
        """Project through PatchMerger, return [total_tokens, llm_dim]."""
        projected = self.projector(feats, gh, gw)  # [B, n_merged, D]

        if patch_mask is not None:
            ms = self.config.spatial_merge_size
            parts = []
            for b in range(projected.shape[0]):
                m2d = patch_mask[b].reshape(gh, gw)
                cm = m2d.reshape(gh // ms, ms, gw // ms, ms).any(1).any(2).flatten()
                parts.append(projected[b][cm])
            return torch.cat(parts, dim=0)

        return projected.reshape(-1, projected.shape[-1])

    def forward(self, input_ids, pixel_values=None, attention_mask=None,
                labels=None, image_token_id=None, patch_attention_mask=None,
                **kwargs):
        if image_token_id is None:
            image_token_id = getattr(self.config, "image_token_id", None) or 151655

        inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            # Cast pixels to LLM dtype (critical for bf16 — matches reference)
            pixel_values = pixel_values.to(dtype=inputs_embeds.dtype)

            feats, gh, gw = self._get_vision_features(pixel_values)
            flat_vis = self._project_and_flatten(feats, gh, gw, patch_attention_mask)

            mask = input_ids == image_token_id
            if mask.sum() > 0:
                bi, si = torch.nonzero(mask, as_tuple=True)
                n = min(len(bi), flat_vis.shape[0])
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[bi[:n], si[:n]] = flat_vis[:n].to(inputs_embeds.dtype)

        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, input_ids, pixel_values=None, attention_mask=None,
                 image_token_id=None, patch_attention_mask=None, **kwargs):
        if image_token_id is None:
            image_token_id = getattr(self.config, "image_token_id", None) or 151655

        inputs_embeds = self.get_input_embeddings()(input_ids)

        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=inputs_embeds.dtype)

            feats, gh, gw = self._get_vision_features(pixel_values)
            flat_vis = self._project_and_flatten(feats, gh, gw, patch_attention_mask)

            mask = input_ids == image_token_id
            if mask.sum() > 0:
                bi, si = torch.nonzero(mask, as_tuple=True)
                n = min(len(bi), flat_vis.shape[0])
                inputs_embeds = inputs_embeds.clone()
                inputs_embeds[bi[:n], si[:n]] = flat_vis[:n].to(inputs_embeds.dtype)

        return self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )