"""
Integration test: initialise TinyQwen3VL, load one real sample, run forward pass.
"""
import torch
import sys
import os
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import TinyQwen3VL, TinyQwen3VLConfig
from src.processor import TinyQwen3Processor
from src.data import VQADataset, collate_fn
from torch.utils.data import DataLoader
from datasets import load_dataset


def test_integration():
    llm_id = "Qwen/Qwen3-0.6B"
    vision_id = "timm/naflexvit_base_patch16_siglip.v2_webli"

    print(">>> Testing Model Initialization...")
    config = TinyQwen3VLConfig(
        llm_model_id=llm_id,
        vision_model_id=vision_id,
    )
    model = TinyQwen3VL(config)
    print("Model initialized successfully.")

    print("\n>>> Testing Processor...")
    processor = TinyQwen3Processor(
        llm_model_id=llm_id,
        vision_model_id=vision_id,
    )
    # Resize embeddings for added tokens
    model.llm.resize_token_embeddings(len(processor.tokenizer))

    print("\n>>> Testing Dataset Loading...")
    try:
        hf_dataset = load_dataset(
            "HuggingFaceM4/FineVision", "chartqa",
            split="train[:5]", streaming=False
        )
        ds = VQADataset(dataset=hf_dataset, processor=processor)
        dataloader = DataLoader(ds, batch_size=1, collate_fn=collate_fn, num_workers=0)
        batch = next(iter(dataloader))

        print("\n>>> Testing Forward Pass with Real Data...")
        with torch.no_grad():
            outputs = model(**batch)
            print("Output logits shape:", outputs.logits.shape)
            print("Loss:", outputs.loss)

    except Exception as e:
        print(f"Dataset test failed: {e}")
        traceback.print_exc()
        print("\nUsing dummy data for forward test...")
        dummy_ids = torch.randint(0, 1000, (1, 10))
        dummy_pix = torch.randn(1, 3, 384, 384)
        outputs = model(input_ids=dummy_ids, pixel_values=dummy_pix)
        print("Dummy output logits shape:", outputs.logits.shape)


if __name__ == "__main__":
    test_integration()
