import torch
import torch.nn as nn
import timm
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PretrainedConfig
)


class TinyQwen3VLConfig(PretrainedConfig):
    """Configuration for TinyQwen3VL following Qwen3VL design patterns."""
    model_type = "tiny_qwen3_vl"

    def __init__(
        self,
        llm_model_id="Qwen/Qwen3-0.6B",
        vision_model_id="timm/naflexvit_base_patch16_siglip.v2_webli",
        freeze_vision=True,
        freeze_llm=False,
        vision_hidden_size=768,
        num_deep_layers=4,
        patch_size=16,
        spatial_merge_size=2,
        image_token_id=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.llm_model_id = llm_model_id
        self.vision_model_id = vision_model_id
        self.freeze_vision = freeze_vision
        self.freeze_llm = freeze_llm
        self.vision_hidden_size = vision_hidden_size
        self.num_deep_layers = num_deep_layers
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.image_token_id = image_token_id


class SpatialMergeProjector(nn.Module):
    """
    Qwen3VL-style 2x2 spatial merge MLP projector.
    Groups 2x2 adjacent vision patches into 1 token (4x reduction),
    then projects concatenated features to LLM dimension via a 2-layer MLP.
    """
    def __init__(self, vision_dim, llm_dim, merge_size=2):
        super().__init__()
        self.merge_size = merge_size
        input_dim = vision_dim * (merge_size ** 2)
        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, llm_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x, grid_h, grid_w):
        """
        Args:
            x: [B, num_patches, vision_dim]
            grid_h, grid_w: patch grid dims (must be divisible by merge_size)
        Returns: [B, num_merged_tokens, llm_dim]
        """
        B, N, D = x.shape
        ms = self.merge_size

        # Reshape to spatial grid -> group 2x2 -> concatenate features
        x = x.reshape(B, grid_h, grid_w, D)
        x = x.reshape(B, grid_h // ms, ms, grid_w // ms, ms, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.reshape(B, (grid_h // ms) * (grid_w // ms), ms * ms * D)

        return self.fc2(self.act(self.fc1(self.norm(x))))


class TinyQwen3VL(PreTrainedModel):
    """
    Tiny Qwen3VL: SigLIP-2 NaFlexViT encoder + Qwen3 LLM decoder
    with 2x2 spatial merge projector and DeepStack injection.
    """
    config_class = TinyQwen3VLConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super().__init__(config)

        # 1. Vision Tower (SigLIP-2 NaFlexViT via timm)
        self.vision_tower = timm.create_model(
            config.vision_model_id,
            pretrained=True,
            dynamic_img_size=True,
            num_classes=0
        )

        # 2. LLM (Qwen3)
        self.llm = AutoModelForCausalLM.from_pretrained(config.llm_model_id)
        llm_dim = self.llm.config.hidden_size

        # 3. Main projector (2x2 spatial merge — replaces simple MLP)
        self.projector = SpatialMergeProjector(
            vision_dim=config.vision_hidden_size,
            llm_dim=llm_dim,
            merge_size=config.spatial_merge_size
        )

        # 4. DeepStack: separate projectors for intermediate ViT layers
        self.num_deep_layers = config.num_deep_layers
        self.deep_projectors = nn.ModuleList([
            SpatialMergeProjector(
                vision_dim=config.vision_hidden_size,
                llm_dim=llm_dim,
                merge_size=config.spatial_merge_size
            )
            for _ in range(config.num_deep_layers)
        ])

        # CRITICAL: Zero-init the output layer of each deep projector so that
        # DeepStack starts as a no-op. Without this, random projections corrupt
        # the LLM hidden states and cause loss >> random chance.
        for proj in self.deep_projectors:
            nn.init.zeros_(proj.fc2.weight)
            nn.init.zeros_(proj.fc2.bias)

        # Freeze as configured
        if config.freeze_vision:
            for p in self.vision_tower.parameters():
                p.requires_grad = False
        if config.freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
        """Delegate gradient checkpointing to the inner LLM."""
        self.llm._set_gradient_checkpointing(enable, gradient_checkpointing_func)

    # ---- Embedding helpers ----
    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.set_input_embeddings(value)

    # ---- Vision feature extraction (multi-layer for DeepStack) ----
    def _extract_vision_features(self, pixel_values):
        """
        Returns:
            final_feats: [B, N, D] from last ViT layer
            intermediates: list of [B, N, D] from evenly-spaced ViT layers
            grid_h, grid_w: patch grid dimensions
        """
        B, C, H, W = pixel_values.shape
        ps = self.config.patch_size
        grid_h, grid_w = H // ps, W // ps

        num_blocks = len(self.vision_tower.blocks)
        # Pick evenly-spaced layers for DeepStack
        indices = [
            int(i * num_blocks / (self.num_deep_layers + 1))
            for i in range(1, self.num_deep_layers + 1)
        ]

        captured = {}
        hooks = []
        for idx in indices:
            def _hook(module, inp, out, layer_idx=idx):
                captured[layer_idx] = out
            hooks.append(self.vision_tower.blocks[idx].register_forward_hook(_hook))

        final_feats = self.vision_tower.forward_features(pixel_values)

        for h in hooks:
            h.remove()

        intermediates = [captured[idx] for idx in indices]
        return final_feats, intermediates, grid_h, grid_w

    # ---- Embedding merge ----
    def _merge_embeddings(self, input_ids, pixel_values, image_token_id,
                          patch_attention_mask=None):
        """
        Replace image-placeholder tokens with 2x2-compressed vision features.
        Returns:
            inputs_embeds, deep_visual_list, image_positions, n_visual
        """
        inputs_embeds = self.get_input_embeddings()(input_ids)
        deep_list = None
        img_pos = None
        n_vis = 0

        if pixel_values is not None:
            final_feats, intermediates, gh, gw = \
                self._extract_vision_features(pixel_values)

            # Project final features (2x2 merge)
            image_features = self.projector(final_feats, gh, gw)

            # Project intermediate features for DeepStack
            deep_list = [
                self.deep_projectors[i](intermediates[i], gh, gw)
                for i in range(len(intermediates))
            ]

            # Handle NaFlex patch mask — downsample for 2x2 merge
            ms = self.config.spatial_merge_size
            if patch_attention_mask is not None:
                valid_lists = []
                deep_valid = [[] for _ in range(len(deep_list))]
                for b in range(image_features.shape[0]):
                    mask2d = patch_attention_mask[b].reshape(gh, gw)
                    cmask = mask2d.reshape(
                        gh // ms, ms, gw // ms, ms
                    ).any(dim=1).any(dim=2).flatten()
                    valid_lists.append(image_features[b][cmask])
                    for k in range(len(deep_list)):
                        deep_valid[k].append(deep_list[k][b][cmask])
                flat_visual = torch.cat(valid_lists, dim=0)
                deep_list = [torch.cat(dv, dim=0) for dv in deep_valid]
            else:
                flat_visual = image_features.reshape(-1, image_features.shape[-1])
                deep_list = [d.reshape(-1, d.shape[-1]) for d in deep_list]

            # Replace placeholder tokens
            image_mask = (input_ids == image_token_id)
            if image_mask.sum() > 0:
                bi, si = torch.nonzero(image_mask, as_tuple=True)
                n_vis = min(bi.shape[0], flat_visual.shape[0])
                bi, si = bi[:n_vis], si[:n_vis]
                img_pos = (bi, si)
                inputs_embeds[bi, si] = flat_visual[:n_vis].to(inputs_embeds.dtype)
                deep_list = [d[:n_vis] for d in deep_list]

        return inputs_embeds, deep_list, img_pos, n_vis

    # ---- Forward (training) with DeepStack hooks ----
    def forward(self, input_ids, pixel_values=None, attention_mask=None,
                labels=None, image_token_id=None, patch_attention_mask=None,
                **kwargs):
        if image_token_id is None:
            image_token_id = getattr(self.config, 'image_token_id', None) or 151665

        embeds, deep_list, img_pos, n_vis = self._merge_embeddings(
            input_ids, pixel_values, image_token_id, patch_attention_mask
        )

        # DeepStack: inject intermediate vision features into early LLM layers
        hooks = []
        if deep_list and img_pos and n_vis > 0:
            inject_idxs = list(range(min(len(deep_list),
                                         len(self.llm.model.layers))))
            seq_len = embeds.shape[1]
            for i, li in enumerate(inject_idxs):
                feats = deep_list[i]
                bi, si = img_pos

                def _make_hook(b, s, f, expected_len):
                    def hook_fn(module, inp, out):
                        hs = out[0]
                        if hs.shape[1] != expected_len:
                            return out  # skip during KV-cache steps
                        hs = hs.clone()
                        hs[b, s] = hs[b, s] + f.to(hs.dtype)
                        return (hs,) + out[1:]
                    return hook_fn

                hooks.append(
                    self.llm.model.layers[li].register_forward_hook(
                        _make_hook(bi, si, feats, seq_len)))

        output = self.llm(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels
        )

        for h in hooks:
            h.remove()

        return output

    # ---- Generate (inference) — no DeepStack for simplicity ----
    @torch.no_grad()
    def generate(self, input_ids, pixel_values=None, attention_mask=None,
                 image_token_id=None, patch_attention_mask=None, **kwargs):
        if image_token_id is None:
            image_token_id = getattr(self.config, 'image_token_id', None) or 151665

        embeds, _, _, _ = self._merge_embeddings(
            input_ids, pixel_values, image_token_id, patch_attention_mask
        )

        return self.llm.generate(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            **kwargs
        )