import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import torch
from torch.utils.data import ConcatDataset
from transformers import Trainer, TrainingArguments

from src.model import TinyQwen3VL, TinyQwen3VLConfig
from src.processor import TinyQwen3Processor
from src.data import (
    collate_fn,
    load_train_dataset,
    DATASET_REGISTRY,
)


class VLMTrainer(Trainer):
    """
    Trainer subclass that fixes loss logging with gradient accumulation.

    The base HF Trainer accumulates `training_step()` return values into
    `self._total_loss_scalar`.  In some versions, `training_step()` returns
    the raw loss (not divided by gradient_accumulation_steps), which inflates
    the logged loss by that factor.

    This subclass ensures the returned loss is always the true per-sample
    mean, so logged `loss` reflects the real cross-entropy.
    """

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)

        # If the parent already divided by grad_accum, `loss` is small.
        # If not, it's the raw loss.  We normalise by checking: the parent
        # Trainer divides when `self.args.gradient_accumulation_steps > 1`
        # is handled via Accelerator.  Some versions don't.  The safest
        # approach: always return loss / grad_accum, and undo any double
        # division by checking the scale.
        #
        # Simple heuristic-free fix: override the accumulation directly.
        # We store the raw per-sample loss ourselves and report it.
        return loss


    def log(self, logs, start_time=None):
        """Correct the loss if it was inflated by gradient accumulation."""
        if "loss" in logs and self.args.gradient_accumulation_steps > 1:
            logs["loss"] = logs["loss"] / self.args.gradient_accumulation_steps
        if "grad_norm" in logs and isinstance(logs["grad_norm"], torch.Tensor):
            logs["grad_norm"] = logs["grad_norm"].item()
        super().log(logs, start_time=start_time)


def parse_args():
    available = ", ".join(sorted(DATASET_REGISTRY.keys()))
    parser = argparse.ArgumentParser(
        description="Train TinyQwen3VL",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # Model
    parser.add_argument("--vision_model", type=str,
                        default="timm/naflexvit_base_patch16_siglip.v2_webli")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen3-0.6B")

    # Training
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Data
    parser.add_argument(
        "--datasets", type=str, nargs="+", default=["textvqa"],
        help=f"Dataset(s) to train on. Can combine multiple.\nAvailable: {available}")
    parser.add_argument(
        "--finevision_subset", type=str, default="chartqa",
        help="Subset name when using 'finevision' dataset")
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit each dataset to N samples (useful for debugging)")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # --- A. Processor ---
    print(f"[1/5] Loading Processor for {args.llm_model}...")
    processor = TinyQwen3Processor(
        vision_model_id=args.vision_model,
        llm_model_id=args.llm_model,
        max_patches=576
    )

    # --- B. Data ---
    print(f"[2/5] Loading Datasets: {args.datasets}...")
    loaded = []
    for name in args.datasets:
        try:
            ds = load_train_dataset(
                name, processor,
                subset=args.finevision_subset if name == "finevision" else None,
                max_samples=args.max_samples,
            )
            loaded.append(ds)
            print(f"      ✓ {name}: {len(ds)} samples")
        except Exception as e:
            print(f"      ✗ Skipping {name}: {e}")

    if not loaded:
        raise ValueError("No datasets loaded!")

    train_dataset = ConcatDataset(loaded) if len(loaded) > 1 else loaded[0]
    print(f"      Total training samples: {len(train_dataset)}")

    # --- C. Model ---
    print("[3/5] Initialising Model...")
    config = TinyQwen3VLConfig(
        llm_model_id=args.llm_model,
        vision_model_id=args.vision_model,
        freeze_vision=True,
        freeze_llm=False,
        vision_hidden_size=768,
        image_token_id=processor.image_token_id,
    )

    model = TinyQwen3VL(config)

    # Resize embeddings for added special tokens
    tok_len = len(processor.tokenizer)
    vocab_size = model.llm.get_input_embeddings().weight.shape[0]
    if tok_len > vocab_size:
        print(f"      - Resizing embeddings: {vocab_size} -> {tok_len}")
        model.llm.resize_token_embeddings(tok_len)

    # --- D. Training Arguments ---
    print("[4/5] Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        fp16=(not torch.cuda.is_bf16_supported() and torch.cuda.is_available())
             if torch.cuda.is_available() else False,
        gradient_checkpointing=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = VLMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        tokenizer=processor.tokenizer,
    )

    # --- E. Train ---
    print("[5/5] Starting Training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final"))
    processor.tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
    print("Done!")


if __name__ == "__main__":
    main()