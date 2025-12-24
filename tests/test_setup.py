import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import NanoQwenVL, NanoQwenVLConfig
from src.data import NanoVLMDataset, collate_fn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import traceback

def test_integration():
    print(">>> Testing Model Initialization...")
    config = NanoQwenVLConfig(
        llm_model_id="Qwen/Qwen3-0.6B",
        vision_model_id="moonshotai/MoonViT-SO-400M",
    )
    model = NanoQwenVL(config)
    print("Model initialized successfully.")

    print("\n>>> Testing Dataset Loading with Streaming...")
    # Test with dummy data instead of real dataset to avoid long downloads
    try:
        print("Initializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
        
        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Add image placeholder token
        if "<|image_pad|>" not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": ["<|image_pad|>"]})
        
        # Resize model embeddings to match tokenizer
        model.llm.resize_token_embeddings(len(tokenizer))
        
        print("Loading streaming dataset (will only fetch 1 sample)...")
        hf_dataset = load_dataset("HuggingFaceM4/FineVision", "CoSyn_400k_document", split="train", streaming=True)
        
        # Initialize NanoVLMDataset with the new API
        ds = NanoVLMDataset(
            dataset=hf_dataset,
            tokenizer=tokenizer
        )
        
        # Get one batch
        dataloader = DataLoader(ds, batch_size=1, collate_fn=collate_fn, num_workers=0)
        batch = next(iter(dataloader))
        
        print("\n>>> Testing Forward Pass with Real Data...")
        with torch.no_grad():
            outputs = model(**batch)
            print("Output logits shape:", outputs.logits.shape)
            print("Loss (if labels present):", outputs.loss)
            
    except Exception as e:
        print(f"Dataset loading failed or skipped: {e}")
        traceback.print_exc()
        # Use dummy data if dataset fails
        print("\nUsing dummy data for model forward test...")
        dummy_input_ids = torch.randint(0, 1000, (1, 10))
        dummy_pixel_values = torch.randn(1, 3, 384, 384)
        outputs = model(input_ids=dummy_input_ids, pixel_values=dummy_pixel_values)
        print("Dummy Output logits shape:", outputs.logits.shape)

if __name__ == "__main__":
    test_integration()
