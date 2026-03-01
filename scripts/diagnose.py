"""Deep diagnostic: check labels, bf16 vs fp32, and simulated training step."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from src.model import TinyQwen3VL, TinyQwen3VLConfig
from src.processor import TinyQwen3Processor
from src.data import load_train_dataset, collate_fn
from torch.utils.data import DataLoader

VISION = "timm/naflexvit_base_patch16_siglip.v2_webli"
LLM = "Qwen/Qwen3-0.6B"

print("=== 1. Setup ===")
processor = TinyQwen3Processor(vision_model_id=VISION, llm_model_id=LLM)
config = TinyQwen3VLConfig(
    llm_model_id=LLM, vision_model_id=VISION,
    image_token_id=processor.image_token_id,
)
model = TinyQwen3VL(config)
model.llm.resize_token_embeddings(len(processor.tokenizer))

print(f"\n=== 2. Load 3 samples ===")
ds = load_train_dataset("textvqa", processor, max_samples=10)
dl = DataLoader(ds, batch_size=1, collate_fn=collate_fn, num_workers=0)

for i, batch in enumerate(dl):
    if i >= 3:
        break
    if not batch:
        print(f"  Sample {i}: EMPTY BATCH (all None)")
        continue

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    n_total = input_ids.shape[1]
    n_active = (labels != -100).sum().item()
    n_img = (input_ids == batch["image_token_id"]).sum().item()

    # Decode the active label tokens to see what we're training on
    active_mask = labels[0] != -100
    active_ids = labels[0][active_mask]
    active_text = processor.tokenizer.decode(active_ids, skip_special_tokens=False)

    # Also decode the full input to check structure
    full_text = processor.tokenizer.decode(input_ids[0], skip_special_tokens=False)

    print(f"\n  --- Sample {i} ---")
    print(f"  Seq len: {n_total}, Image tokens: {n_img}, Active labels: {n_active}")
    print(f"  Active text (what model trains on): '{active_text}'")
    print(f"  Full prompt (first 200 chars): '{full_text[:200]}...'")

    # Check loss in fp32
    model.eval()
    with torch.no_grad():
        out = model(**batch)
        print(f"  Loss (fp32 eval): {out.loss.item():.4f}")

    # Check loss in train mode fp32
    model.train()
    with torch.no_grad():
        out_train = model(**batch)
        print(f"  Loss (fp32 train): {out_train.loss.item():.4f}")

    # Check loss in bf16
    if torch.cuda.is_available():
        model_gpu = model.to("cuda")
        batch_gpu = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out_bf16 = model_gpu(**batch_gpu)
            print(f"  Loss (bf16 autocast): {out_bf16.loss.item():.4f}")
        model_gpu = model_gpu.to("cpu")
    else:
        print(f"  (No CUDA — skipping bf16 test)")

    # Simulated training step (fp32, single micro-batch)
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-5
    )
    optimizer.zero_grad()
    out_step = model(**batch)
    print(f"  Loss (fp32 train step): {out_step.loss.item():.4f}")
    out_step.loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(f"  Grad norm: {grad_norm.item():.4f}")
    optimizer.step()
    optimizer.zero_grad()

print("\n=== Done ===")
