import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, AutoImageProcessor
from PIL import Image
import numpy as np

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
VISION_CONFIG = {
    "model_id": "moonshotai/MoonViT-SO-400M",
    "vision_dim": 1152,
}

class NanoVLMDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, vision_config=VISION_CONFIG):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.vision_config = vision_config
        
        # Load MoonViT image processor
        print(f"Loading Image Processor for {vision_config['model_id']}...")
        self.image_processor = AutoImageProcessor.from_pretrained(
            vision_config['model_id'], 
            trust_remote_code=True
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
            "texts": [{"user": "...", "assistant": "..."}, ...],
            ...
        }
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
            
            # Process image with MoonViT processor
            # Returns pixel_values and image_grid_hws
            image_inputs = self.image_processor([image], return_tensors="pt")
            # Keep pixel_values with batch dim for proper batching later
            pixel_values = image_inputs['pixel_values']  # [1, C, H, W] or similar
            # image_grid_hws is a tensor of shape [N, 2] where N = num images
            image_grid_hws = image_inputs['image_grid_hws']  # Tensor[N, 2]

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
            # For MoonViT, we don't inject image placeholders into text
            # Instead, image embeddings are prepended in the model's forward pass
            # So we just tokenize the conversation normally
            
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
            
            # Iterate through turns (user -> assistant -> user -> assistant)
            # We assume the list is ordered: human, gpt, human, gpt...
            for idx, msg in enumerate(conversations):
                role = msg.get('from', '').lower()
                content = msg.get('value', '').strip()
                
                if role in ['human', 'user']:
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
                "pixel_values": pixel_values,
                "image_grid_hws": image_grid_hws,
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
# Data Collator (Handles variable-length image tokens)
# ---------------------------------------------------------
def collate_fn(batch):
    # Filter out any Nones
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return {}

    # Concatenate pixel values along batch dim
    # Each item has pixel_values of shape [1, ...] or [C, H, W]
    pixel_values_list = []
    for item in batch:
        pv = item['pixel_values']
        if pv.dim() == 3:  # [C, H, W] -> [1, C, H, W]
            pv = pv.unsqueeze(0)
        pixel_values_list.append(pv)
    pixel_values = torch.cat(pixel_values_list, dim=0)
    
    # Concatenate image_grid_hws tensors
    # Each item has image_grid_hws of shape [N, 2]
    image_grid_hws_list = []
    for item in batch:
        hws = item['image_grid_hws']
        if isinstance(hws, torch.Tensor):
            image_grid_hws_list.append(hws)
        elif isinstance(hws, list) and len(hws) > 0:
            # Convert list of tuples to tensor
            image_grid_hws_list.append(torch.tensor(hws))
    
    if image_grid_hws_list:
        image_grid_hws = torch.cat(image_grid_hws_list, dim=0)
    else:
        image_grid_hws = None
    
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
        "image_grid_hws": image_grid_hws,
        "attention_mask": (padded_input_ids != 0).long()
    }