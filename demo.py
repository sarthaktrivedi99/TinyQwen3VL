"""
Gradio demo for NanoQwenVL - Vision Language Model
"""
import argparse
import gradio as gr
import torch
from PIL import Image
from transformers import AutoTokenizer
from torchvision import transforms
import timm
from timm.data import resolve_model_data_config
from peft import PeftModel

from src.model import NanoQwenVL, NanoQwenVLConfig


def load_model(checkpoint_path=None, lora_path=None):
    """Load the model (optionally with full checkpoint or LoRA adapter)."""
    import os
    
    print("Loading base model...")
    
    config = NanoQwenVLConfig(
        llm_model_id="google/gemma-3-270m-it",
        vision_model_id="naflexvit_base_patch16_siglip.v2_webli",
        freeze_vision=True,
        freeze_llm=True
    )
    
    model = NanoQwenVL(config)
    
    if checkpoint_path:
        # Load full checkpoint (all weights)
        print(f"Loading full checkpoint from {checkpoint_path}...")
        if os.path.isdir(checkpoint_path):
            # Directory with pytorch_model.bin or model.safetensors
            import glob
            ckpt_file = None
            for pattern in ["pytorch_model.bin", "model.safetensors", "*.pt", "*.pth"]:
                matches = glob.glob(os.path.join(checkpoint_path, pattern))
                if matches:
                    ckpt_file = matches[0]
                    break
            if ckpt_file:
                state_dict = torch.load(ckpt_file, map_location="cpu")
                model.load_state_dict(state_dict, strict=False)
                print(f"Loaded checkpoint from {ckpt_file}")
            else:
                print(f"Warning: No checkpoint file found in {checkpoint_path}")
        else:
            # Single file
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint from {checkpoint_path}")
    
    elif lora_path:
        # Load LoRA adapter
        print(f"Loading LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        model = model.merge_and_unload()  # Merge for faster inference
        print("LoRA adapter loaded and merged.")
        
        # Also load projector weights (saved separately)
        projector_path = os.path.join(lora_path, "projector.pt")
        if os.path.exists(projector_path):
            print(f"Loading projector from {projector_path}...")
            projector_state = torch.load(projector_path, map_location="cpu")
            model.projector.load_state_dict(projector_state)
            print("Projector loaded successfully.")
        else:
            print(f"Warning: No projector.pt found at {projector_path}. Projector is still randomly initialized!")
    
    model.eval()
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    return model, device


def create_image_transform():
    """Create image transform for PE-Core."""
    temp_model = timm.create_model("naflexvit_base_patch16_siglip.v2_webli", pretrained=False)
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


# Global variables (initialized in main)
MODEL = None
DEVICE = None  
TOKENIZER = None
IMAGE_TRANSFORM = None


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
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}"


def create_demo():
    """Create the Gradio interface."""
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
    
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NanoQwenVL Gradio Demo")
    parser.add_argument("--checkpoint_path", type=str, default=None, 
                        help="Path to full model checkpoint (directory or file)")
    parser.add_argument("--lora_path", type=str, default=None, 
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--share", action="store_true", 
                        help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the demo on")
    args = parser.parse_args()
    
    # Initialize model and tokenizer
    print("=" * 60)
    print("Initializing NanoQwenVL Demo...")
    print("=" * 60)
    
    MODEL, DEVICE = load_model(checkpoint_path=args.checkpoint_path, lora_path=args.lora_path)
    TOKENIZER = AutoTokenizer.from_pretrained("google/gemma-3-270m-it", trust_remote_code=True)
    IMAGE_TRANSFORM = create_image_transform()
    
    # Create and launch demo
    demo = create_demo()
    demo.launch(share=args.share, server_name="0.0.0.0", server_port=args.port)
