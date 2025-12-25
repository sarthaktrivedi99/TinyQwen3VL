"""
Gradio demo for NanoQwenVL - Vision Language Model
"""
import gradio as gr
import torch
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
import timm
from timm.data import resolve_model_data_config

from src.model import NanoQwenVL, NanoQwenVLConfig


def load_model(checkpoint_path=None):
    """Load the model (optionally from a checkpoint)."""
    print("Loading model...")
    
    config = NanoQwenVLConfig(
        llm_model_id="Qwen/Qwen3-0.6B",
        vision_model_id="vit_pe_core_small_patch16_384.fb",
        freeze_vision=True,
        freeze_llm=True
    )
    
    model = NanoQwenVL(config)
    
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}...")
        # For PEFT/LoRA checkpoints, you'd use different loading
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    return model, device


def create_image_transform():
    """Create image transform for PE-Core."""
    temp_model = timm.create_model("vit_pe_core_small_patch16_384.fb", pretrained=False)
    data_config = resolve_model_data_config(temp_model)
    mean = data_config.get('mean', (0.5, 0.5, 0.5))
    std = data_config.get('std', (0.5, 0.5, 0.5))
    
    def pad_to_patch_size(img, patch_size=16):
        w, h = img.size
        new_w = ((w + patch_size - 1) // patch_size) * patch_size
        new_h = ((h + patch_size - 1) // patch_size) * patch_size
        if new_w != w or new_h != h:
            padded = Image.new('RGB', (new_w, new_h), (0, 0, 0))
            padded.paste(img, (0, 0))
            return padded
        return img
    
    transform = transforms.Compose([
        transforms.Lambda(lambda img: pad_to_patch_size(img, 16)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform


# Initialize globally
print("Initializing NanoQwenVL Demo...")
MODEL, DEVICE = load_model()
TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
IMAGE_TRANSFORM = create_image_transform()


def generate_response(image, text, max_new_tokens=256, temperature=0.7, top_p=0.9):
    """Generate a response given an image and text prompt."""
    if image is None:
        return "Please upload an image."
    if not text.strip():
        return "Please enter a text prompt."
    
    try:
        # Convert to PIL if needed
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert('RGB')
        
        # Process image
        pixel_values = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        
        # Process text
        inputs = TOKENIZER(text, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)
        
        # Generate
        with torch.no_grad():
            outputs = MODEL.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=TOKENIZER.pad_token_id or TOKENIZER.eos_token_id,
            )
        
        # Decode response
        response = TOKENIZER.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input text from response if present
        if response.startswith(text):
            response = response[len(text):].strip()
        
        return response
    
    except Exception as e:
        return f"Error: {str(e)}"


# Create Gradio interface
with gr.Blocks(title="NanoQwenVL Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔮 NanoQwenVL Demo
    A compact Vision-Language Model combining PE-Core vision encoder with Qwen3 LLM.
    
    Upload an image and enter a prompt to get a response!
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="Upload Image", type="pil")
            text_input = gr.Textbox(
                label="Text Prompt", 
                placeholder="Describe this image...",
                lines=3
            )
            
            with gr.Row():
                max_tokens = gr.Slider(
                    minimum=32, maximum=512, value=256, step=32,
                    label="Max New Tokens"
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1,
                    label="Temperature"
                )
            
            submit_btn = gr.Button("🚀 Generate", variant="primary")
        
        with gr.Column(scale=1):
            output = gr.Textbox(label="Model Response", lines=12)
    
    # Example prompts
    gr.Examples(
        examples=[
            ["What do you see in this image?"],
            ["Describe the main objects in this image."],
            ["What is happening in this scene?"],
            ["Read any text visible in this image."],
        ],
        inputs=[text_input],
    )
    
    # Connect the button
    submit_btn.click(
        fn=generate_response,
        inputs=[image_input, text_input, max_tokens, temperature],
        outputs=output
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
