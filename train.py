import os
import torch
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from src.model import NanoQwenVL, NanoQwenVLConfig
from src.data import NanoVLMDataset, collate_fn

def train():
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
        vision_model_id="hf-hub:timm/PE-Core-S-16-384",
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
    print("Applying LoRA...")
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=16, 
        lora_alpha=32, 
        lora_dropout=0.1,
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
    training_args = TrainingArguments(
        output_dir="./nano_qwen_vl_checkpoints",
        per_device_train_batch_size=2, 
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        max_steps=10000, 
        logging_steps=10, # 1 is too noisy for long runs
        save_steps=500,
        fp16=True, # Highly recommended to set True for VRAM savings if using GPU
        push_to_hub=False,
        remove_unused_columns=False, # Essential for custom VLM datasets
        report_to="none",
        gradient_checkpointing=False, # Disabled by user request
        dataloader_pin_memory=True,
        dataloader_num_workers=4,  # Parallel data loading to prevent blocking
        dataloader_prefetch_factor=2  # Prefetch 2 batches per worker
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
    trainer.save_model("./nano_qwen_vl_final")
    print("Training finished and model saved.")

if __name__ == "__main__":
    train()