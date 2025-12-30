"""
Dataloader verification test for TinyQwen3VL.
"""
import torch
import sys
import os
from torch.utils.data import DataLoader
from datasets import load_dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.processor import TinyQwen3Processor
from src.data import VQADataset, collate_fn


def test_dataloader():
    llm_id = "Qwen/Qwen3-0.6B"
    vision_id = "timm/naflexvit_base_patch16_siglip.v2_webli"

    print(">>> 1. Initializing Processor...")
    processor = TinyQwen3Processor(
        llm_model_id=llm_id,
        vision_model_id=vision_id,
    )

    print(">>> 2. Loading Dataset (first 10 samples)...")
    hf_dataset = load_dataset(
        "HuggingFaceM4/FineVision", "chartqa",
        split="train[:10]", streaming=False
    )
    dataset = VQADataset(dataset=hf_dataset, processor=processor)

    print(">>> 3. Creating DataLoader...")
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, num_workers=0)

    print(">>> 4. Inspecting Batches...")
    batch = next(iter(dataloader))

    required_keys = {'input_ids', 'labels', 'pixel_values',
                     'attention_mask', 'patch_attention_mask'}
    print(f"Batch keys: {batch.keys()}")
    assert required_keys.issubset(batch.keys()), \
        f"Missing keys: {required_keys - set(batch.keys())}"

    print(f"  input_ids shape:           {batch['input_ids'].shape}")
    print(f"  labels shape:              {batch['labels'].shape}")
    print(f"  pixel_values shape:        {batch['pixel_values'].shape}")
    print(f"  attention_mask shape:      {batch['attention_mask'].shape}")
    print(f"  patch_attention_mask shape: {batch['patch_attention_mask'].shape}")

    assert batch['pixel_values'].dim() == 4, "pixel_values should be 4D [B,C,H,W]"

    # NaFlex: dims divisible by 32 (merge_unit), but variable across batches
    _, _, h, w = batch['pixel_values'].shape
    assert h % 32 == 0 and w % 32 == 0, \
        f"Image dims ({h},{w}) must be divisible by 32 for 2x2 merge"

    # Masking check
    input_ids = batch['input_ids']
    labels = batch['labels']
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids[0])
    label_ids = labels[0].tolist()

    masked = sum(1 for l in label_ids if l == -100)
    active = sum(1 for l in label_ids if l != -100)
    print(f"\n  Masked tokens: {masked}  |  Active tokens: {active}")

    if label_ids[-1] == -100 and label_ids[-2] == -100:
        print("  ⚠️  WARNING: end of sequence is masked — answer may be missing")
    else:
        print("  ✅  Answer tokens are active (unmasked)")

    # Image token check
    img_tok_id = batch['image_token_id']
    img_count = (input_ids == img_tok_id).sum().item()
    print(f"  Image pad tokens in input: {img_count}")
    print("  ✅  All checks passed")


if __name__ == "__main__":
    test_dataloader()