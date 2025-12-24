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
        
        # Load vision transform upfront to avoid lazy loading issues
        print(f"Loading Vision Transforms for {vision_config['model_id']}...")
        _, _, self.transform = open_clip.create_model_and_transforms(
            vision_config['model_id'], 
            pretrained=None
        )
        
        # Setup Special Token
        # If <|image_pad|> isn't found, fallback to unk_token (but ideally it should be added in train.py)
        if "<|image_pad|>" not in tokenizer.get_vocab():
            self.img_token_id = tokenizer.unk_token_id
            self.img_token_str = tokenizer.unk_token
            print(f"[DEBUG] <|image_pad|> not found, using unk_token: {self.img_token_str} ({self.img_token_id})")
        else:
            self.img_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
            self.img_token_str = "<|image_pad|>"
            print(f"[DEBUG] Using <|image_pad|>: {self.img_token_str} ({self.img_token_id})") 

    def process_item(self, item):
        """
        Processes a single item from HuggingFaceM4/FineVision (CoSyn_400k_document).
        Format:
        {
            "images": [PIL.Image, ...],
            "texts": ["Text content...", ...],
            ...
        }
        We treat this as a document parsing task:
        User: <|image_pad|>
        Assistant: <text>
        """
        try:
            # --- A. Image Handling ---
            # FineVision uses 'images' list
            images = item.get('images')
            if not images or len(images) == 0:
                return None
            
            image = images[0]
            
            # Ensure PIL Image
            if not isinstance(image, Image.Image):
                if isinstance(image, str):
                    try:
                        image = Image.open(image).convert('RGB')
                    except Exception as e:
                        return None
                else:
                    return None
            else:
                image = image.convert('RGB')
            
            # Apply Vision Transform -> [3, 384, 384]
            pixel_values = self.transform(image) 

            # --- B. Text Handling ---
            # FineVision uses 'texts' list of dicts: [{'user': '...', 'assistant': '...'}]
            texts = item.get('texts')
            if not texts or len(texts) == 0:
                return None
                
            # Construct conversation from all text turns
            conversations = []
            if isinstance(texts, list):
                for t in texts:
                    if isinstance(t, dict):
                        if 'user' in t and 'assistant' in t:
                            conversations.append({"from": "human", "value": t['user']})
                            conversations.append({"from": "gpt", "value": t['assistant']})
            
            if not conversations:
                return None
            
            # --- C. Tokenization & Masking ---
            image_placeholders = self.img_token_str * self.vision_config['num_tokens']
            
            # ChatML delimiters
            im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
            im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            
            # Tokenize newline (may be multiple tokens)
            nl_ids = self.tokenizer("\n", add_special_tokens=False).input_ids
            
            # Pre-tokenized role headers
            user_header = [im_start] + self.tokenizer("user", add_special_tokens=False).input_ids + nl_ids
            assistant_header = [im_start] + self.tokenizer("assistant", add_special_tokens=False).input_ids + nl_ids
            
            input_ids = []
            labels = []
            
            # Inject image into first user turn
            first_user_turn = True
            
            # Iterate through turns (user -> assistant -> user -> assistant)
            # We assume the list is ordered: human, gpt, human, gpt...
            for idx, msg in enumerate(conversations):
                role = msg.get('from', '').lower()
                content = msg.get('value', '').strip()
                
                if role in ['human', 'user']:
                    # Inject image in first human turn
                    if first_user_turn:
                        content = f"{image_placeholders}\n{content}"
                        first_user_turn = False
                        
                    content_ids = self.tokenizer(content, add_special_tokens=False).input_ids
                    # User turn: Mask everything
                    turn_ids = user_header + content_ids + [im_end] + nl_ids
                    turn_labels = [-100] * len(turn_ids)
                    
                    input_ids.extend(turn_ids)
                    labels.extend(turn_labels)
                    
                elif role in ['gpt', 'assistant']:
                    content_ids = self.tokenizer(content, add_special_tokens=False).input_ids
                    
                    # Assistant turn: Mask header, train on content
                    header_len = len(assistant_header)
                    turn_ids = assistant_header + content_ids + [im_end] + nl_ids
                    
                    turn_labels = (
                        [-100] * header_len +
                        content_ids +
                        [im_end] + nl_ids
                    )
                    
                    input_ids.extend(turn_ids)
                    labels.extend(turn_labels)

            # Truncate
            max_len = 1024
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
                
            if len(input_ids) == 0:
                return None
                
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "pixel_values": pixel_values
            }

        except Exception as e:
            print(f"      [DEBUG] Exception in process_item: {e}")
            import traceback
            traceback.print_exc()
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