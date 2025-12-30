import os
import gradio as gr
import torch
from PIL import Image
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file

from src.model import TinyQwen3VL, TinyQwen3VLConfig
from src.processor import TinyQwen3Processor

# --- Register custom model with Auto classes ---
try:
    AutoConfig.register("tiny_qwen3_vl", TinyQwen3VLConfig)
    AutoModelForCausalLM.register(TinyQwen3VLConfig, TinyQwen3VL)
except ValueError:
    pass  # Already registered

# --- Configuration ---
CHECKPOINT_PATH = "./checkpoints/final/"
LLM_MODEL_ID = "Qwen/Qwen3-0.6B"
VISION_MODEL_ID = "timm/naflexvit_base_patch16_siglip.v2_webli"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading from {CHECKPOINT_PATH} on {DEVICE}...")

# --- Load Processor ---
processor = TinyQwen3Processor(
    llm_model_id=LLM_MODEL_ID,
    vision_model_id=VISION_MODEL_ID,
)

# --- Load Model ---
config = TinyQwen3VLConfig.from_pretrained(CHECKPOINT_PATH, local_files_only=True)
model = TinyQwen3VL(config)

# Resize embeddings to match tokenizer (includes vision tokens)
model.llm.resize_token_embeddings(len(processor.tokenizer))

# Load weights
safetensors_path = os.path.join(CHECKPOINT_PATH, "model.safetensors")
bin_path = os.path.join(CHECKPOINT_PATH, "pytorch_model.bin")

if os.path.exists(safetensors_path):
    state_dict = load_file(safetensors_path)
elif os.path.exists(bin_path):
    state_dict = torch.load(bin_path, map_location="cpu")
else:
    raise FileNotFoundError(f"No model weights found in {CHECKPOINT_PATH}")

model.load_state_dict(state_dict, strict=False)
model.to(DEVICE)
model.eval()
print("Model loaded successfully!")


def inference(image, text, max_new_tokens, temperature):
    if image is None:
        return "Please upload an image."
    if not text:
        text = "Describe this image."

    print(f"\n--- Inference: {text} ---")

    messages = [
        {"role": "user",
         "content": [{"type": "image"}, {"type": "text", "text": text}]}
    ]

    inputs = processor.process(images=image, text=messages, return_tensors="pt")

    print(f"[DEBUG] Input IDs shape: {inputs['input_ids'].shape}")
    print(f"[DEBUG] Pixel values shape: {inputs['pixel_values'].shape}")
    print(f"[DEBUG] Visual tokens (post 2x2 merge): {inputs.get('num_visual_tokens', 'N/A')}")

    pixel_values = inputs["pixel_values"].to(DEVICE)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    generated_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        image_token_id=inputs["image_token_id"],
        max_new_tokens=max_new_tokens,
        min_new_tokens=2,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    input_len = input_ids.shape[1]
    new_tokens = (generated_ids[0][input_len:]
                  if generated_ids.shape[1] > input_len
                  else generated_ids[0])

    generated_text = processor.tokenizer.decode(new_tokens, skip_special_tokens=True)
    print(f"[DEBUG] Output: '{generated_text}'")
    return generated_text


# --- Gradio UI ---
with gr.Blocks(title="TinyQwen3VL Demo") as demo:
    gr.Markdown("# 🔮 TinyQwen3VL Demo")
    gr.Markdown(f"Loaded from: `{CHECKPOINT_PATH}`")

    with gr.Row():
        with gr.Column():
            img_input = gr.Image(type="pil", label="Upload Image")
            txt_input = gr.Textbox(
                label="Question", placeholder="Describe this image.",
                value="Describe this image.")

            with gr.Accordion("Generation Settings", open=True):
                max_tokens = gr.Slider(10, 512, value=128, label="Max New Tokens")
                temp = gr.Slider(0.1, 1.5, value=0.7, label="Temperature")

            btn = gr.Button("Generate", variant="primary")

        with gr.Column():
            output = gr.Textbox(label="Model Output", lines=5)

    btn.click(inference,
              inputs=[img_input, txt_input, max_tokens, temp],
              outputs=output)

if __name__ == "__main__":
    demo.launch(share=True)
