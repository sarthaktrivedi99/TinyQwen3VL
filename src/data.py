"""
Dataset implementation for TinyQwen3VL - Vision Language Model training.
Supports both indexed and iterable datasets with quality filtering.
"""
import torch
import logging
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
import timm
from timm.data import resolve_model_data_config
from torchvision import transforms

# ---------------------------------------------------------
# Vision Model Configuration (NaFlexViT Base SigLIP)
# ---------------------------------------------------------
VISION_CONFIG = {
    "model_id": "naflexvit_base_patch16_siglip.v2_webli",
    "vision_dim": 768,  # NaFlexViT Base embedding dimension
}

# Gemma 3 uses 256 soft tokens per image
NUM_IMAGE_TOKENS = 256
IMAGE_TOKEN = "<start_of_image>"


def get_image_string(tokenizer, num_images=1):
    """Generate image token string for Gemma 3 (256 tokens per image)."""
    return IMAGE_TOKEN * NUM_IMAGE_TOKENS * num_images


class ImageProcessor:
    """Image processor for NaFlexViT with dynamic resolution."""
    
    def __init__(self, vision_config=VISION_CONFIG, max_resolution=None):
        self.vision_config = vision_config
        self.max_resolution = max_resolution
        
        # Get normalization stats from TIMM
        temp_model = timm.create_model(vision_config['model_id'], pretrained=False)
        data_config = resolve_model_data_config(temp_model)
        self.mean = data_config.get('mean', (0.5, 0.5, 0.5))
        self.std = data_config.get('std', (0.5, 0.5, 0.5))
        
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.patch_size = 16
        
        if max_resolution:
            print(f"[ImageProcessor] Max resolution: {max_resolution}")
    
    def set_max_resolution(self, max_resolution):
        """Update max resolution for curriculum learning."""
        self.max_resolution = max_resolution
        if max_resolution:
            print(f"[Resolution Curriculum] Set max resolution to {max_resolution}")
        else:
            print("[Resolution Curriculum] Switched to native resolution")
    
    def _resize_if_needed(self, img):
        """Resize image if max_resolution is set, preserving aspect ratio."""
        if self.max_resolution is None:
            return img
        
        max_size = self.max_resolution
        w, h = img.size
        
        if h > max_size or w > max_size:
            scale = min(max_size / h, max_size / w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
        
        return img
    
    def _pad_to_patch_size(self, img):
        """Pad image to make dimensions divisible by patch_size."""
        w, h = img.size
        new_w = ((w + self.patch_size - 1) // self.patch_size) * self.patch_size
        new_h = ((h + self.patch_size - 1) // self.patch_size) * self.patch_size
        
        if new_w != w or new_h != h:
            padded = Image.new('RGB', (new_w, new_h), (0, 0, 0))
            padded.paste(img, (0, 0))
            return padded
        return img
    
    def __call__(self, img):
        """Process image and return tensor with grid size info."""
        img = self._resize_if_needed(img)
        img = self._pad_to_patch_size(img)
        
        w, h = img.size
        grid_h = h // self.patch_size
        grid_w = w // self.patch_size
        
        tensor = self.to_tensor(img)
        tensor = self.normalize(tensor)
        
        return tensor, (grid_h, grid_w)


class BaseDataset(Dataset):
    """Base dataset class with quality filtering and chat template support."""
    
    def __init__(
        self,
        dataset,
        tokenizer,
        image_processor,
        mp_image_token_length=1,
        relevance_min_rating=1,
        image_correspondence_min_rating=1,
        visual_dependency_min_rating=1,
        formatting_min_rating=1
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mp_image_token_length = mp_image_token_length
        self.relevance_min_rating = relevance_min_rating
        self.image_correspondence_min_rating = image_correspondence_min_rating
        self.visual_dependency_min_rating = visual_dependency_min_rating
        self.formatting_min_rating = formatting_min_rating
        
        # Set up Gemma 3 image token if not present
        if IMAGE_TOKEN not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})
        self.image_token = IMAGE_TOKEN
        self.image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
        
        self.prefix_len = self._get_prefix_len()

    def __len__(self):
        return len(self.dataset)

    def _get_prefix_len(self):
        """Calculate the prefix length for loss masking."""
        try:
            random_string = "xzyvd"
            random_string_templated = self.tokenizer.apply_chat_template(
                [{"role": "assistant", "content": random_string}],
                tokenize=False,
                add_special_tokens=False
            )
            random_string_location = random_string_templated.find(random_string)
            return len(self.tokenizer.encode(random_string_templated[:random_string_location]))
        except Exception:
            return 0

    def _get_messages(self, item, num_images=0):
        \"\"\"Extract and filter messages from item.\"\"\"
        messages = []
        texts = item.get('texts', [])
        
        for index, text in enumerate(texts):
            # Quality filtering
            try:
                if item.get('relevance_ratings') and item['relevance_ratings'][index] is not None:
                    if item['relevance_ratings'][index] < self.relevance_min_rating:
                        continue
                if item.get('image_correspondence_ratings') and item['image_correspondence_ratings'][index] is not None:
                    if item['image_correspondence_ratings'][index] < self.image_correspondence_min_rating:
                        continue
                if item.get('visual_dependency_ratings') and item['visual_dependency_ratings'][index] is not None:
                    if item['visual_dependency_ratings'][index] < self.visual_dependency_min_rating:
                        continue
                if item.get('formatting_ratings') and item['formatting_ratings'][index] is not None:
                    if item['formatting_ratings'][index] < self.formatting_min_rating:
                        continue
            except Exception as e:
                logging.warning(f"Error processing item ratings: {e}")

            if isinstance(text, dict):
                messages.append({"role": "user", "content": text.get('user', '')})
                messages.append({"role": "assistant", "content": text.get('assistant', '')})

        if len(messages) == 0:
            return messages

        # Safety: remove any existing image tokens
        for msg in messages:
            if self.image_token in msg["content"]:
                msg["content"] = msg["content"].replace(self.image_token, "")

        # Prepend image tokens to first message (256 tokens per image)
        if num_images > 0:
            image_string = get_image_string(self.tokenizer, num_images)
            messages[0]["content"] = image_string + messages[0]["content"]

        return messages

    def _process_images(self, images):
        """Process list of PIL images."""
        processed_images = []
        splitted_image_counts = []
        
        for image in images:
            if isinstance(image, Image.Image):
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                processed_image, grid_size = self.image_processor(image)
                processed_images.append(processed_image)
                splitted_image_counts.append(grid_size)
            else:
                logging.warning(f"Skipping non-PIL image: {type(image)}")
        
        return processed_images, splitted_image_counts

    def _prepare_inputs_and_loss_mask(self, messages):
        """Prepare input_ids and loss mask using chat template."""
        conv_result = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            return_dict=True,
        )
        input_ids = list(conv_result["input_ids"])
        
        # Add EOS token at the end if not already present
        eos_token_id = self.tokenizer.eos_token_id
        if eos_token_id is not None and (len(input_ids) == 0 or input_ids[-1] != eos_token_id):
            input_ids.append(eos_token_id)
        
        mask = [0] * len(input_ids)
        
        # For Gemma and similar models, we identify assistant tokens by:
        # 1. Finding the encoded assistant content in the full sequence
        # 2. Marking those positions as trainable
        
        for msg in messages:
            if msg["role"] == "assistant":
                content = msg["content"]
                if content:
                    # Encode just the content
                    content_ids = self.tokenizer.encode(content, add_special_tokens=False)
                    content_len = len(content_ids)
                    
                    # Search for this content in the full sequence
                    for i in range(len(input_ids) - content_len + 1):
                        if input_ids[i:i+content_len] == content_ids:
                            mask[i:i+content_len] = [1] * content_len
                            break
        
        # Also train on EOS token (important for generation to know when to stop)
        if eos_token_id is not None and len(input_ids) > 0 and input_ids[-1] == eos_token_id:
            mask[-1] = 1

        return (
            torch.tensor(input_ids),
            torch.tensor(mask).to(torch.bool),
            torch.tensor([1] * len(input_ids))  # All tokens are attended to
        )


class VQADataset(BaseDataset):
    """Visual Question Answering Dataset with indexed access."""

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return self._process_data(item)

    def _process_data(self, item):
        """Process a single data item."""
        # Handle images
        images_data = item.get('images')
        if images_data is None:
            images_data = []
        elif not isinstance(images_data, list):
            images_data = [images_data]

        processed_images = []
        num_images = 0
        if images_data:
            processed_images, _ = self._process_images(images_data)
            num_images = len(processed_images)

        messages = self._get_messages(item, num_images)

        if len(messages) == 0:
            return None

        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)

        # Stack images if present
        if processed_images:
            pixel_values = torch.stack(processed_images) if len(processed_images) > 1 else processed_images[0].unsqueeze(0)
        else:
            pixel_values = None

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image_token_id": self.image_token_id,
        }

    def _get_labels(self, input_ids, mask):
        """Create labels with -100 for non-assistant tokens (causal LM style)."""
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)  # Shift labels for causal LM
        labels[-1] = -100  # Last token has no target
        return labels


class VQAIterableDataset(IterableDataset, BaseDataset):
    """Iterable version of VQADataset for streaming large datasets."""
    
    def __init__(self, *args, **kwargs):
        BaseDataset.__init__(self, *args, **kwargs)
    
    def __iter__(self):
        for item in self.dataset:
            result = self._process_data(item)
            if result is not None:
                yield result
    
    def _process_data(self, item):
        """Process a single data item (same as VQADataset)."""
        images_data = item.get('images')
        if images_data is None:
            images_data = []
        elif not isinstance(images_data, list):
            images_data = [images_data]

        processed_images = []
        num_images = 0
        if images_data:
            processed_images, _ = self._process_images(images_data)
            num_images = len(processed_images)

        messages = self._get_messages(item, num_images)

        if len(messages) == 0:
            return None

        input_ids, mask, attention_mask = self._prepare_inputs_and_loss_mask(messages)
        labels = self._get_labels(input_ids, mask)

        if processed_images:
            pixel_values = torch.stack(processed_images) if len(processed_images) > 1 else processed_images[0].unsqueeze(0)
        else:
            pixel_values = None

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "image_token_id": self.image_token_id,
        }
    
    def _get_labels(self, input_ids, mask):
        labels = input_ids.clone().masked_fill(~mask, -100)
        labels = labels.roll(-1)
        labels[-1] = -100
        return labels


def collate_fn(batch):
    """Collate function for DataLoader - handles variable-sized images."""
    import torch.nn.functional as F
    
    # Filter Nones
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}

    # Handle variable-sized images (NaFlex produces different sizes)
    pixel_values_list = [item['pixel_values'] for item in batch if item['pixel_values'] is not None]
    
    if pixel_values_list:
        # Squeeze any extra batch dims and get max dimensions
        processed = []
        for pv in pixel_values_list:
            if pv.dim() == 4:  # [B, C, H, W]
                pv = pv.squeeze(0)  # [C, H, W]
            processed.append(pv)
        
        max_h = max(pv.shape[1] for pv in processed)
        max_w = max(pv.shape[2] for pv in processed)
        
        # Pad each image to max dimensions
        padded_pixel_values = []
        for pv in processed:
            c, h, w = pv.shape
            pad_h, pad_w = max_h - h, max_w - w
            padded = F.pad(pv, (0, pad_w, 0, pad_h), value=0)
            padded_pixel_values.append(padded)
        
        pixel_values = torch.stack(padded_pixel_values)
    else:
        pixel_values = None

    # Pad input_ids, attention_mask, and labels
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    result = {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels,
    }
    
    if pixel_values is not None:
        result["pixel_values"] = pixel_values
    
    # Get image_token_id from first item (convert to tensor for device compatibility)
    if batch and isinstance(batch[0], dict) and 'image_token_id' in batch[0]:
        result["image_token_id"] = torch.tensor(batch[0]["image_token_id"])
    
    return result


# Backwards compatibility alias
NanoVLMDataset = VQAIterableDataset