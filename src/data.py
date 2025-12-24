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
        Supports ShareGPT format:
        {
            "image": "path/to/img" or url,
            "conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "..."}
            ]
        }
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

            # --- B. Parse Conversations (ShareGPT) ---
            # Fallback for pure text datasets or simple Q&A keys if strictly needed, 
            # but we assume ShareGPT format as primary now.
            conversations = item.get('conversations', [])
            if not conversations:
                # Minimal fallback if user didn't update data yet to avoid total crash
                if 'question' in item and 'answer' in item:
                    conversations = [
                        {"from": "human", "value": item['question']},
                        {"from": "gpt", "value": item['answer']}
                    ]
                else:
                    return None

            # --- C. Tokenization & Masking (Multi-turn) ---
            image_placeholders = self.img_token_str * self.vision_config['num_tokens']
            
            input_ids = []
            labels = []
            
            # ChatML delimiters
            im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
            im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            nl = self.tokenizer.convert_tokens_to_ids("\n")
            
            # Pre-tokenized role headers
            user_header = [im_start] + self.tokenizer("user", add_special_tokens=False).input_ids + [nl]
            assistant_header = [im_start] + self.tokenizer("assistant", add_special_tokens=False).input_ids + [nl]
            
            # Inject image into first user turn
            first_user_turn = True
            
            for msg in conversations:
                role = msg.get('from', '').lower()
                content = msg.get('value', '').strip()
                
                # Check for image placeholder manually if needed, or auto-inject
                # We auto-inject at start of first user message
                if role in ['human', 'user']:
                    if first_user_turn:
                        content = f"{image_placeholders}\n{content}"
                        first_user_turn = False
                        
                    # Tokenize content
                    content_ids = self.tokenizer(content, add_special_tokens=False).input_ids
                    
                    # Full turn: Header + Content + End
                    turn_ids = user_header + content_ids + [im_end, nl]
                    
                    # Mask user turn in labels
                    turn_labels = [-100] * len(turn_ids)
                    
                elif role in ['gpt', 'chatgpt', 'assistant', 'model']:
                    # Tokenize content
                    content_ids = self.tokenizer(content, add_special_tokens=False).input_ids
                    
                    # Full turn: Header + Content + End
                    turn_ids = assistant_header + content_ids + [im_end, nl] # Note: usually \n after end
                    
                    # Train on assistant content only. 
                    # Mask header? Usually yes.
                    # Mask end? Usually no (predict EOS).
                    
                    header_len = len(assistant_header)
                    content_len = len(content_ids)
                    end_len = 2 # <|im_end|>\n
                    
                    # Mask header (-100), train on content + end tokens
                    turn_labels = (
                        [-100] * header_len + 
                        content_ids + 
                        [im_end, nl]
                    )
                else:
                    continue # Skip unknown roles (e.g. system)

                input_ids.extend(turn_ids)
                labels.extend(turn_labels)

            # Truncate
            max_len = 1024
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                labels = labels[:max_len]
                
            # Drop if empty
            if len(input_ids) == 0:
                return None
                
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