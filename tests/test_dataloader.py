import torch
import sys
import os
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from PIL import Image

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data import NanoVLMDataset, collate_fn  

def inverse_normalize(tensor, mean, std):
    """Reverses the normalization applied by the transform for visualization."""
    # tensor: (3, H, W)
    mean = torch.as_tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.as_tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return tensor * std + mean

def test_dataloader_manual_verification():
    print(">>> 1. Initializing Tokenizer...")
    
    tokenizer_id = "Qwen/Qwen2.5-0.5B-Instruct" 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(">>> 2. Initializing Dataset...")
    # Download a small subset instead of streaming to avoid hanging
    # We take just the first 10 samples for testing
    print("   Downloading first 10 samples from dataset (this may take a moment)...")
    hf_dataset = load_dataset(
        "HuggingFaceM4/FineVision", 
        "CoSyn_400k_document", 
        split="train[:10]",  # Only download first 10 samples
        streaming=False  # Force download instead of streaming
    )
    
    # Initialize your NanoVLMDataset (which is now an IterableDataset)
    dataset = NanoVLMDataset(
        dataset=hf_dataset, 
        tokenizer=tokenizer
    )

    print(">>> 3. Creating DataLoader...")
    # Note: num_workers=0 is safer for simple debugging of streams
    dataloader = DataLoader(
        dataset, 
        batch_size=2, 
        collate_fn=collate_fn,
        num_workers=0 
    )

    print(">>> 4. Inspecting Batches...")
    
    # Standard OpenCLIP Mean/Std
    OPENCLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    OPENCLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    # We cannot use enumerate(dataloader) easily to skip, so we just grab the first one
    # or iterate normally.
    try:
        data_iter = iter(dataloader)
        batch = next(data_iter)
    except StopIteration:
        print("❌ Error: Dataset yielded no items!")
        return
    except Exception as e:
        print(f"❌ Error fetching batch: {e}")
        import traceback
        traceback.print_exc()
        return

    print(f"\n=== Batch Analysis ===")
    input_ids = batch['input_ids']
    labels = batch['labels']
    pixel_values = batch['pixel_values']
    
    print(f"Tensor Shapes:")
    print(f"  Input IDs:    {input_ids.shape}")
    print(f"  Labels:       {labels.shape}")
    print(f"  Pixel Values: {pixel_values.shape} (Expect B, 3, 384, 384)")

    # --- VERIFICATION A: Image Placeholders ---
    # We need to find the ID used for image padding. 
    # Since we can't access dataset.img_token_id easily if not stored, we assume the dataset set it up.
    # We'll use the ID from the dataset object if accessible, or infer it.
    placeholder_id = dataset.img_token_id 
    
    token_count = (input_ids[0] == placeholder_id).sum().item()
    print(f"\n[Check] Image Tokens per sample: {token_count}")
    if token_count == 576:
        print("  ✅ SUCCESS: Exactly 576 tokens found.")
    else:
        print(f"  ❌ FAILURE: Found {token_count} tokens. Expected 576.")

    # --- VERIFICATION B: Text & Masking ---
    print("\n[Check] Decoding Input vs. Labels (Masking Check):")
    
    decoded_input = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    
    # Visualizing Masking
    full_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    label_tokens_ids = labels[0].tolist()
    
    reconstructed_label_view = []
    # Safety check for length mismatch (shouldn't happen)
    for i, label_id in enumerate(label_tokens_ids):
        if i >= len(full_tokens): break
        token = full_tokens[i]
        
        if label_id == -100:
            reconstructed_label_view.append("[MASKED]")
        else:
            reconstructed_label_view.append(token)
    
    print("-" * 60)
    print("FULL INPUT (First 300 chars):")
    print(decoded_input[:300].replace('\n', '\\n') + " ...")
    print("-" * 60)
    print("TRAINING TARGETS (Last 300 tokens):")
    # Join and print the end to see the answer
    print(" ".join(reconstructed_label_view)[-300:]) 
    print("-" * 60)
    
    # Check if the end is masked (It shouldn't be)
    if label_tokens_ids[-1] == -100 and label_tokens_ids[-2] == -100:
            print("  ⚠️ WARNING: The end of the sequence seems to be masked. Is the answer missing?")
    else:
            print("  ✅ SUCCESS: The answer (end of seq) is active/unmasked.")

    # --- VERIFICATION C: Image Visuals ---
    print(f"\n[Check] Saving Image Sample...")
    img_tensor = pixel_values[0].cpu()  # 3, H, W
    
    # Un-normalize
    img_tensor = inverse_normalize(img_tensor, OPENCLIP_MEAN, OPENCLIP_STD)
    img_tensor = torch.clamp(img_tensor, 0, 1)
    img_tensor = (img_tensor * 255).byte()
    
    img = Image.fromarray(img_tensor.permute(1, 2, 0).numpy())
    filename = "debug_stream_sample.png"
    img.save(filename)
    print(f"  ✅ Saved image to {filename}.")

if __name__ == "__main__":
    test_dataloader_manual_verification()