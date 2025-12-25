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


def test_dataloader_manual_verification():
    print(">>> 1. Initializing Tokenizer...")
    
    tokenizer_id = "google/gemma-3-270m-it" 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=True)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Add image placeholder token
    if "<|image_pad|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|image_pad|>"]})
    
    print("\n>>> Verifying token count...")
    # PE-Core with NaFlex produces variable token count based on image size
    print("  With PE-Core NaFlex, token count varies with input image resolution")
    
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
    # Inspect the first batch
    print("\n>>> Inspecting first batch...")
    required_keys = {'input_ids', 'labels', 'pixel_values', 'attention_mask'}
    print(f"Batch keys: {batch.keys()}")
    assert required_keys.issubset(batch.keys()), f"Missing keys: {required_keys - set(batch.keys())}"
    
    print(f"  input_ids shape: {batch['input_ids'].shape}")
    print(f"  labels shape: {batch['labels'].shape}")
    print(f"  pixel_values shape: {batch['pixel_values'].shape}")
    print(f"  attention_mask shape: {batch['attention_mask'].shape}")
    
    # PE-Core expects fixed [B, C, H, W] pixel values
    assert batch['pixel_values'].dim() == 4, "pixel_values should be 4D [B, C, H, W]"
    assert batch['pixel_values'].shape[1:] == (3, 384, 384), "Expected 3x384x384 images with PE-Core"

    print(f"\n=== Batch Analysis ===")
    input_ids = batch['input_ids']
    labels = batch['labels']
    pixel_values = batch['pixel_values']
    image_grid_hws = batch.get('image_grid_hws')
    
    print(f"Tensor Shapes:")
    print(f"  Input IDs:    {input_ids.shape}")
    print(f"  Labels:       {labels.shape}")
    print(f"  Pixel Values: {pixel_values.shape}")
    if image_grid_hws is not None:
        print(f"  Image Grid HWs: {image_grid_hws}")

    # --- VERIFICATION A: Check batch has required keys ---
    print("\n[Check] Batch contents:")
    print(f"  Keys: {list(batch.keys())}")
    print("  ✅ SUCCESS: Batch contains all required keys.")

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

    # --- VERIFICATION C: Image Info ---
    print(f"\n[Check] Image processed successfully")
    print(f"  ✅ Pixel values shape: {pixel_values.shape}")
    if image_grid_hws is not None:
        print(f"  ✅ Image grid HWs provided (MoonViT native resolution)")

# NaFlexViT Configuration
VISION_CONFIG = {
    "model_id": "naflexvit_base_patch16_siglip.v2_webli",
    "vision_dim": 768,  # NaFlexViT Base embedding dimension
}

if __name__ == "__main__":
    test_dataloader_manual_verification()