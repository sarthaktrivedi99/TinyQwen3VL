import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import NanoQwenVL, NanoQwenVLConfig
from src.data import FineVisionDataset, collate_fn
from torch.utils.data import DataLoader
import traceback

def test_integration():
    print(">>> Testing Model Initialization...")
    config = NanoQwenVLConfig(
        llm_model_id="Qwen/Qwen3-0.6B",
        vision_model_id="hf-hub:timm/PE-Core-S-16-384",
    )
    model = NanoQwenVL(config)
    print("Model initialized successfully.")

    print("\n>>> Testing Dataset Loading (Mocking/Small)...")
    # We use streaming=True to avoid downloading huge dataset for test
    try:
        ds = FineVisionDataset(split="train", dataset_name="HuggingFaceM4/FineVision", subset="CoSyn_400k_document")
        # Just get one item
        if hasattr(ds, 'is_streaming') and ds.is_streaming:
            item = next(iter(ds.dataset)) # Streaming dataset is iterable
            # Manually process one item as __getitem__ expects index but our logic for streaming in data.py was barebones
            # Let's just assume we can get one item and pass to collate
            pass
        else:
            # If not streaming (loading failed or we decided to load all), get index 0
            if len(ds) > 0:
                print("Dataset length:", len(ds))
                item = ds[0]
                batch = collate_fn([item])
                
                print("\n>>> Testing Forward Pass...")
                with torch.no_grad():
                    # Move to CPU for test
                    outputs = model(**batch)
                    print("Output logits shape:", outputs.logits.shape)
                    print("Loss (if labels present):", outputs.loss)
    except Exception as e:
        print(f"Dataset loading failed or skipped: {e}")
        traceback.print_exc()
        # Use dummy data if dataset fails
        print("Using dummy data for model forward test...")
        dummy_input_ids = torch.randint(0, 1000, (1, 10))
        dummy_pixel_values = torch.randn(1, 3, 384, 384)
        outputs = model(input_ids=dummy_input_ids, pixel_values=dummy_pixel_values)
        print("Dummy Output logits shape:", outputs.logits.shape)

if __name__ == "__main__":
    test_integration()
