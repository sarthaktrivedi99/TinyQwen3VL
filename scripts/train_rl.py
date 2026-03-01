import os
import sys
import argparse
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig

from src.model import TinyQwen3VL, TinyQwen3VLConfig
from src.processor import TinyQwen3Processor

def format_reward_func(completions, ground_truth=None, **kwargs):
    """Simple Exact Match Reward"""
    rewards = []
    for comp, gt in zip(completions, ground_truth):
        # Depending on trl version, comp might be a list of messages or a string
        if isinstance(comp, list) and len(comp) > 0 and isinstance(comp[-1], dict):
            gen_text = comp[-1].get("content", "").strip().lower()
        else:
            gen_text = str(comp).strip().lower()
        
        gt_lower = str(gt).strip().lower()
        if gt_lower in gen_text:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def format_dataset_for_grpo(dataset_name="textvqa", split="train[:500]"):
    # Load dataset
    hf_id = "howard-hou/OCR-VQA" if dataset_name == "ocrvqa" else "facebook/textvqa"
    ds = load_dataset(hf_id, split=split, trust_remote_code=True)
    
    def format_row(row):
        question = row.get("question", "")
        # Get answers properly
        if "answers" in row:
            ans = row["answers"]
            ans = ans[0] if isinstance(ans, list) and len(ans) > 0 else ans
        else:
            ans = ""
            
        return {
            "prompt": [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}
            ],
            "image": row.get("image"),
            "ground_truth": ans
        }
        
    ds = ds.map(format_row, remove_columns=ds.column_names, load_from_cache_file=False)
    # Filter out empty images
    ds = ds.filter(lambda x: x["image"] is not None)
    return ds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision_model", type=str, default="timm/naflexvit_base_patch16_siglip.v2_webli")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_rl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/4] Loading Processor...")
    processor = TinyQwen3Processor(
        vision_model_id=args.vision_model,
        llm_model_id=args.llm_model,
    )

    print("[2/4] Formatting Dataset for GRPO...")
    train_dataset = format_dataset_for_grpo("textvqa", split="train[:500]")
    
    print(f"Dataset size: {len(train_dataset)}")

    print("[3/4] Initializing Model...")
    config = TinyQwen3VLConfig(
        llm_model_id=args.llm_model,
        vision_model_id=args.vision_model,
        freeze_vision=True,
        freeze_llm=False,
        image_token_id=processor.image_token_id,
    )
    model = TinyQwen3VL(config)
    
    # Resize embeddings for special tokens if needed
    tok_len = len(processor.tokenizer)
    vocab_size = model.llm.get_input_embeddings().weight.shape[0]
    if tok_len > vocab_size:
        print(f"Resizing embeddings: {vocab_size} -> {tok_len}")
        model.llm.resize_token_embeddings(tok_len)

    print("[4/4] Setting up GRPOTrainer...")
    
    # For GRPOTrainer vLLM integration is off by default when it cannot be imported or false is set.
    # Our custom model might not be natively supported by vLLM, so we stick to HF generate.
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_generations=2,
        max_prompt_length=256,
        max_completion_length=64,
        max_steps=args.max_steps,
        save_steps=50,
        logging_steps=5,
        use_vllm=False,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=(not torch.cuda.is_bf16_supported() and torch.cuda.is_available()) if torch.cuda.is_available() else False,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=format_reward_func,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=processor,
    )

    print("Starting RL Training with GRPO...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    processor.tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    print("Done!")

if __name__ == "__main__":
    main()
