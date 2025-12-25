import os
import argparse
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer, TrainerCallback
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from src.model import NanoQwenVL, NanoQwenVLConfig
from src.data import VQAIterableDataset, ImageProcessor, collate_fn


class ResolutionCurriculumCallback(TrainerCallback):
    """Callback to switch from low resolution to native resolution during training."""
    
    def __init__(self, image_processor, switch_step=2000):
        self.image_processor = image_processor
        self.switch_step = switch_step
        self.switched = False
    
    def on_step_begin(self, args, state, control, **kwargs):
        if not self.switched and state.global_step >= self.switch_step:
            print(f"\n{'='*60}")
            print(f"[Resolution Curriculum] Step {state.global_step}: Switching to native resolution")
            print(f"{'='*60}\n")
            self.image_processor.set_max_resolution(None)  # Native resolution
            self.switched = True


class ProjectorSaveCallback(TrainerCallback):
    """Callback to save projector weights at every checkpoint (since LoRA doesn't save it)."""
    
    def __init__(self, model, output_dir):
        self.model = model
        self.output_dir = output_dir
    
    def on_save(self, args, state, control, **kwargs):
        # Save projector to the checkpoint directory
        checkpoint_dir = f"{self.output_dir}/checkpoint-{state.global_step}"
        projector_path = f"{checkpoint_dir}/projector.pt"
        
        # Access projector through PEFT wrapper
        try:
            projector = self.model.base_model.model.projector
        except AttributeError:
            projector = self.model.projector
        
        torch.save(projector.state_dict(), projector_path)
        print(f"[ProjectorSaveCallback] Saved projector to {projector_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Train NanoQwenVL model")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (default: 32)")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA for training (default: True)")
    parser.add_argument("--no_lora", action="store_true", help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--lora_vision", action="store_true", help="Enable LoRA for vision backbone (default: False)")
    
    # DeepSpeed arguments
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to DeepSpeed config JSON file")
    
    # Precision arguments
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision (recommended for A100/H100)")
    parser.add_argument("--fp16", action="store_true", default=True, help="Use float16 precision (default: True)")
    parser.add_argument("--no_fp16", action="store_true", help="Disable fp16, use fp32")
    parser.add_argument("--flash_attention", action="store_true", help="Use Flash Attention 2 (requires flash-attn package)")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./nano_qwen_vl_checkpoints", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size (default: 2)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64, help="Gradient accumulation steps (default: 64)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--max_steps", type=int, default=10000, help="Max training steps (default: 10000)")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every N steps (default: 50)")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps (default: 10)")
    
    # Resolution curriculum arguments
    parser.add_argument("--low_res", type=int, default=448, help="Low resolution for curriculum (default: 448)")
    parser.add_argument("--res_switch_step", type=int, default=2000, help="Step to switch to native resolution (default: 2000)")
    parser.add_argument("--no_curriculum", action="store_true", help="Disable resolution curriculum")
    
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
    llm_model_id = "google/gemma-3-270m-it"
    print(f"Loading tokenizer: {llm_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(llm_model_id, trust_remote_code=True)
    
    # Fix padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # CRITICAL: Add the Gemma 3 image token
    from src.data import IMAGE_TOKEN
    if IMAGE_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [IMAGE_TOKEN]})

    # ------------------------------------------------------------------
    # 2. Model Initialization
    # ------------------------------------------------------------------
    print("Initialize Model...")
    config = NanoQwenVLConfig(
        llm_model_id=llm_model_id,
        vision_model_id="naflexvit_base_patch16_siglip.v2_webli",
        freeze_vision=True, 
        freeze_llm=False,
        use_flash_attention=args.flash_attention
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
        
        # LLM target modules (Gemma architecture)
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        # Add vision backbone modules if enabled (using regex to target vision_tower specifically)
        if args.lora_vision:
            print("  -> Including vision backbone in LoRA training")
            # Use regex patterns to specifically target vision_tower modules
            # This avoids conflicts with LLM modules that have similar names
            target_modules.extend([
                r"vision_tower\.blocks\.\d+\.attn\.qkv",     # Attention QKV
                r"vision_tower\.blocks\.\d+\.attn\.proj",    # Attention output projection
                r"vision_tower\.blocks\.\d+\.mlp\.fc1",      # MLP first layer
                r"vision_tower\.blocks\.\d+\.mlp\.fc2",      # MLP second layer
            ])
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=args.lora_r, 
            lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            # NOTE: modules_to_save removed due to DeepSpeed ZeRO compatibility issues
        )
        
        # This wraps the model. Base LLM weights -> Frozen. LoRA adapters -> Trainable.
        model = get_peft_model(model, peft_config)
        
        # Manually ensure projector is trainable (DeepSpeed compatible approach)
        # The projector is new/random and MUST be trained
        for param in model.base_model.model.projector.parameters():
            param.requires_grad = True
        
        model.print_trainable_parameters()
    else:
        print("Training without LoRA (full fine-tuning)...")
    
    # ------------------------------------------------------------------
    # 4. Dataset Setup
    # ------------------------------------------------------------------
    print("Initialize Dataset...")
    
    # Load the raw HF data first
    # Using streaming to avoid massive RAM usage
    raw_dataset = load_dataset("HuggingFaceM4/FineVision", "cocoqa", split="train", streaming=True)
    
    # Create image processor with resolution curriculum
    initial_max_res = None if args.no_curriculum else args.low_res
    image_processor = ImageProcessor(max_resolution=initial_max_res)
    
    # Create dataset with new VQAIterableDataset
    train_dataset = VQAIterableDataset(
        dataset=raw_dataset, 
        tokenizer=tokenizer,
        image_processor=image_processor,
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
        dataloader_num_workers=4,  # Parallel data loading
        dataloader_prefetch_factor=2,  # Prefetch 2 batches per worker
        accelerator_config={"dispatch_batches": False},  # Required for IterableDataset
        deepspeed=args.deepspeed,  # DeepSpeed config path (None if not used)
    )
    
    # ------------------------------------------------------------------
    # 6. Trainer Execution
    # ------------------------------------------------------------------
    # Callbacks
    callbacks = []
    if not args.no_curriculum:
        print(f"Resolution curriculum: low res={args.low_res} -> native at step {args.res_switch_step}")
        callbacks.append(ResolutionCurriculumCallback(image_processor, switch_step=args.res_switch_step))
    
    # Add projector save callback when using LoRA (projector not saved by PEFT)
    if use_lora:
        callbacks.append(ProjectorSaveCallback(model, args.output_dir))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collate_fn,
        callbacks=callbacks
    )
    
    print("Starting training...")
    trainer.train()
    
    # 7. Save final model
    trainer.save_model(f"{args.output_dir}/final")
    
    # CRITICAL: Also save projector weights separately (not saved by LoRA)
    if use_lora:
        projector_path = f"{args.output_dir}/final/projector.pt"
        torch.save(model.base_model.model.projector.state_dict(), projector_path)
        print(f"Projector saved to {projector_path}")
    
    print("Training finished and model saved.")

if __name__ == "__main__":
    train()