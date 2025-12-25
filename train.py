import os
import argparse
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from src.model import NanoQwenVL, NanoQwenVLConfig
from src.data import NanoVLMDataset, collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description="Train NanoQwenVL model")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA for training (default: True)")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA (full fine-tuning)")
    
    # DeepSpeed arguments
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config JSON file")
    
    # Precision arguments
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision (recommended for A100/H100)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use float16 precision (default: True)")
    parser.add_argument("--no_fp16", action="store_true", help="Disable fp16, use fp32")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./nano_qwen_vl_checkpoints", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size (default: 2)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64, help="Gradient accumulation steps (default: 64)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--max_steps", type=int, default=10000, help="Max training steps (default: 10000)")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every N steps (default: 50)")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps (default: 10)")
    
    return parser.parse_args()

def train():
    args = parse_args()
    
    # Handle conflicting args
    use_lora = args.use_lora and not args.no_lora
    use_bf16 = args.bf16
    use_fp16 = args.fp16 and not args.no_fp16 and not args.bf16  # bf16 takes precedence
    
    # ------------------------------------------------------------------
    # 1. Tokenizer Setup (Must be done first to resize model)
    # ------------------------------------------------------------------
    llm_model_id = "Qwen/Qwen3-0.6B" # Kept exactly as requested
    print(f"Loading tokenizer: {llm_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_id, trust_remote_code=True)
    
    # Fix padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # CRITICAL: Add the image placeholder token
    # This prevents the "token alignment" issues discussed earlier
    if "<|image_pad|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|image_pad|>"]})

    # ------------------------------------------------------------------
    # 2. Model Initialization
    # ------------------------------------------------------------------
    print("Initialize Model...")
    config = NanoQwenVLConfig(
        llm_model_id=llm_model_id,
        vision_model_id="vit_pe_core_small_patch16_384.fb",
        freeze_vision=True, 
        freeze_llm=False # We let PEFT handle the freezing
    )
    
    model = NanoQwenVL(config)
    
    # CRITICAL: Resize embeddings to fit the new <|image_pad|> token
    # We assume 'model.llm' is the attribute holding the QwenForCausalLM
    model.llm.resize_token_embeddings(len(tokenizer))
    
    # ------------------------------------------------------------------
    # 3. Apply LoRA (And save Projector)
    # ------------------------------------------------------------------
    if use_lora:
        print(f"Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout})...")
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            # Standard Qwen target modules
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            
            # CRITICAL: The projector is new/random and MUST be trained.
            # "modules_to_save" ensures these layers remain trainable and are saved 
            # in the adapter checkpoint, not ignored.
            # Check your src.model source code to ensure the attribute is named 'projector'.
            modules_to_save=["projector"] 
        )
        
        # This wraps the model. Base LLM weights -> Frozen. LoRA + Projector -> Trainable.
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print("Training without LoRA (full fine-tuning)...")
    
    # ------------------------------------------------------------------
    # 4. Dataset Setup
    # ------------------------------------------------------------------
    print("Initialize Dataset...")
    
    # Load the raw HF data first
    # Using streaming to avoid massive RAM usage, or standard load if dataset is small
    raw_dataset = load_dataset("HuggingFaceM4/FineVision", "CoSyn_400k_document", split="train", streaming=True)
    
    # Use the NanoVLMDataset wrapper (from previous steps)
    # We must pass the tokenizer so it uses the correct <|image_pad|> ID
    train_dataset = NanoVLMDataset(
        dataset=raw_dataset, 
        tokenizer=tokenizer
    )
    
    # ------------------------------------------------------------------
    # 5. Training Arguments
    # ------------------------------------------------------------------
    print(f"Training config: bf16={use_bf16}, fp16={use_fp16}, deepspeed={args.deepspeed}")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size, 
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps, 
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        push_to_hub=False,
        remove_unused_columns=False, # Essential for custom VLM datasets
        report_to="none",
        gradient_checkpointing=False, # Disabled by user request
        dataloader_pin_memory=True,
        dataloader_num_workers=4,  # Parallel data loading to prevent blocking
        dataloader_prefetch_factor=2,  # Prefetch 2 batches per worker
        deepspeed=args.deepspeed,  # DeepSpeed config path (None if not used)
    )
    
    # ------------------------------------------------------------------
    # 6. Trainer Execution
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn
    )
    
    print("Starting training...")
    trainer.train()
    
    # 7. Save
    trainer.save_model(f"{args.output_dir}/final")
    print("Training finished and model saved.")

if __name__ == "__main__":
    train()