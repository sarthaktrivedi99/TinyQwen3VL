import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer
import open_clip
from PIL import Image
import numpy as np

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
VISION_CONFIG = {
    "model_id": "hf-hub:timm/PE-Core-S-16-384",
    "image_size": 384,
    "patch_size": 16,
    "num_tokens": (384 // 16) ** 2  # 576 tokens
}

class NanoVLMDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, vision_config=VISION_CONFIG):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.vision_config = vision_config
        self._transform = None  # Lazy initialization for multiprocessing efficiency
        
        # Setup Special Token
        # If <|image_pad|> isn't found, fallback to unk_token (but ideally it should be added in train.py)
        if "<|image_pad|>" not in tokenizer.get_vocab():
            self.img_token_id = tokenizer.unk_token_id
            self.img_token_str = tokenizer.unk_token
        else:
            self.img_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            self.img_token_str = "<|image_pad|>"
    
    @property
    def transform(self):
        """Lazy load vision transform to reduce worker initialization overhead."""
        if self._transform is None:
            print(f"Loading Vision Transforms for {self.vision_config['model_id']}...")
            _, _, self._transform = open_clip.create_model_and_transforms(
                self.vision_config['model_id'], 
                pretrained=None
            )
        return self._transform 

    def process_item(self, item):
        """
        Processes a single raw item into a tensor dictionary.
        """
        try:
            # --- A. Image Handling ---
            image = item.get('image') or (item.get('images')[0] if item.get('images') else None)
            
            # Skip corrupted/missing images
            if image is None: 
                return None
                
            if isinstance(image, str):
                try:
                    image = Image.open(image).convert('RGB')
                except:
                    return None # Skip bad paths
            
            # Apply Vision Transform -> [3, 384, 384]
            pixel_values = self.transform(image) 

            # --- B. Text Handling ---
            image_placeholders = self.img_token_str * self.vision_config['num_tokens']
            
            # Extract Q&A (Generic robust extractor)
            question = "Describe this document." 
            answer = "Content not available."
            
            if 'question' in item and 'answer' in item:
                question = item['question']
                answer = item['answer']
            elif 'texts' in item and isinstance(item['texts'], list) and len(item['texts']) > 0:
                t = item['texts'][0] 
                question = t.get('user', question)
                answer = t.get('assistant', answer)

            # ChatML Formatting
            # Structure: <|im_start|>user\n <image_tokens> \n Question <|im_end|>\n <|im_start|>assistant\n Answer <|im_end|>
            user_text = f"<|im_start|>user\n{image_placeholders}\n{question}<|im_end|>\n<|im_start|>assistant\n"
            assistant_text = f"{answer}<|im_end|>"
            
            # --- C. Tokenization ---
            user_tokens = self.tokenizer(user_text, add_special_tokens=False).input_ids
            assistant_tokens = self.tokenizer(assistant_text, add_special_tokens=False).input_ids
            
            input_ids = user_tokens + assistant_tokens
            labels = [-100] * len(user_tokens) + assistant_tokens
            
            # Truncate
            max_len = 1024
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
                
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "pixel_values": pixel_values
            }
        except Exception as e:
            return None

    def __iter__(self):
        """
        Yields items one by one from the stream.
        """
        for item in self.dataset:
            processed = self.process_item(item)
            if processed is not None:
                yield processed

# ---------------------------------------------------------
# Data Collator (Standard Padding)
# ---------------------------------------------------------
def collate_fn(batch):
    # Filter out any Nones that might have slipped through (though __iter__ handles it mostly)
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}

    # Stack images
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # Pad input_ids and labels
    # padding_value for input_ids is 0 (or tokenizer.pad_token_id)
    # padding_value for labels is -100 (ignore index)
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        "input_ids": padded_input_ids,
        "labels": padded_labels,
        "pixel_values": pixel_values,
        "attention_mask": (padded_input_ids != 0).long()
    }