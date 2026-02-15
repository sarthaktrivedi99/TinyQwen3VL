import torch
import math
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms


class TinyQwen3Processor:
    """
    Processor for TinyQwen3VL.
    - Uses Qwen3 tokenizer with added vision tokens (<|vision_start|>, etc.)
    - NaFlex dynamic resolution with 2x2-merge-aware snapping (dims divisible by 32)
    - Computes post-compression token count for placeholder insertion
    """

    # Qwen-style vision special tokens
    VISION_START = "<|vision_start|>"
    VISION_END   = "<|vision_end|>"
    IMAGE_PAD    = "<|image_pad|>"

    def __init__(self, vision_model_id, llm_model_id,
                 max_patches=576, patch_size=16, spatial_merge_size=2):
        print(f"[Processor] Loading Tokenizer: {llm_model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_id)

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Register vision special tokens
        special_tokens = [self.VISION_START, self.VISION_END, self.IMAGE_PAD]
        num_added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_tokens}
        )
        print(f"[Processor] Added {num_added} special tokens")

        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.IMAGE_PAD)
        print(f"[Processor] Image pad token '{self.IMAGE_PAD}' → ID {self.image_token_id}")

        # Vision config
        self.max_patches = max_patches
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.merge_unit = patch_size * spatial_merge_size  # 32

        # SigLIP normalization + NaFlex resize (snap to merge_unit for even grids)
        self.transform = transforms.Compose([
            NaFlexResize(max_patches=max_patches,
                         patch_size=patch_size,
                         spatial_merge_size=spatial_merge_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def _build_image_placeholder(self, num_merged_tokens):
        """Build <|vision_start|><|image_pad|>*N<|vision_end|> string."""
        return (
            self.VISION_START
            + self.IMAGE_PAD * num_merged_tokens
            + self.VISION_END
        )

    def process(self, images=None, text=None, return_tensors="pt",
                add_generation_prompt=True):
        result = {}
        vision_token_count = 0  # after 2x2 merge

        # ---- Process Image ----
        if images is not None:
            if not isinstance(images, list):
                images = [images]

            pixel_values_list = []
            for img in images:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                pixel_values_list.append(self.transform(img))

            pixel_values = pixel_values_list[0].unsqueeze(0)
            _, _, h, w = pixel_values.shape

            grid_h = h // self.patch_size
            grid_w = w // self.patch_size
            num_patches = grid_h * grid_w
            ms = self.spatial_merge_size
            vision_token_count = (grid_h // ms) * (grid_w // ms)

            result["pixel_values"] = pixel_values
            result["num_patches"] = num_patches
            result["num_visual_tokens"] = vision_token_count

        # ---- Process Text ----
        if text is not None:
            # text is a list of message dicts, possibly with {type: "image"} parts
            processed_msgs = []
            for msg in text:
                content = msg.get("content", "")
                if isinstance(content, str):
                    processed_msgs.append(msg)
                    continue

                # Content is a list of parts
                text_content = ""
                for part in content:
                    if part["type"] == "image":
                        text_content += self._build_image_placeholder(vision_token_count)
                    elif part["type"] == "text":
                        text_content += part["text"]

                processed_msgs.append({"role": msg["role"], "content": text_content})

            prompt = self.tokenizer.apply_chat_template(
                processed_msgs, tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

            text_inputs = self.tokenizer(
                prompt,
                return_tensors=return_tensors,
                padding="longest",
                truncation=False,
                max_length=2048
            )

            result["input_ids"] = text_inputs["input_ids"]
            result["attention_mask"] = text_inputs["attention_mask"]

        result["image_token_id"] = self.image_token_id
        return result


class NaFlexResize:
    """
    Resize image to fit within a patch budget while:
    - Preserving aspect ratio
    - Ensuring output dims are divisible by merge_unit (patch_size * spatial_merge_size)
      so that 2x2 spatial merge always gets even grid dimensions.
    """
    def __init__(self, max_patches=576, patch_size=16, spatial_merge_size=2):
        self.max_patches = max_patches
        self.patch_size = patch_size
        self.merge_unit = patch_size * spatial_merge_size  # 32

    def __call__(self, img):
        w, h = img.size
        target_area = self.max_patches * (self.patch_size ** 2)
        scale = math.sqrt(target_area / (w * h))

        new_w = int(round(w * scale / self.merge_unit) * self.merge_unit)
        new_h = int(round(h * scale / self.merge_unit) * self.merge_unit)

        # At least one merge unit
        new_w = max(new_w, self.merge_unit)
        new_h = max(new_h, self.merge_unit)

        # Don't exceed budget
        if (new_w * new_h) // (self.patch_size ** 2) > self.max_patches:
            if new_w > new_h:
                new_w -= self.merge_unit
            else:
                new_h -= self.merge_unit

        return img.resize((new_w, new_h), resample=Image.BICUBIC)