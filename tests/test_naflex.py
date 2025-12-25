"""
Test to verify PE-Core with NaFlex produces variable token counts.
"""
import torch
import timm
from timm.data import resolve_model_data_config, create_transform
from PIL import Image
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_naflex_variable_tokens():
    """Test that NaFlexViT SigLIP produces different token counts for different image sizes."""
    
    print("=" * 60)
    print("Testing NaFlexViT SigLIP Variable Token Count")
    print("=" * 60)
    
    # Load NaFlexViT model with NaFlex
    model_id = "naflexvit_base_patch16_siglip.v2_webli"
    print(f"\n>>> Loading {model_id}...")
    
    model = timm.create_model(
        model_id,
        pretrained=True,
        use_naflex=True,
        dynamic_img_size=True, 
        dynamic_img_pad=True,
        # Note: Don't set num_classes=0 or global_pool='' for SigLIP models
        # as they cause weight loading issues. Use forward_features() instead.
    )
    model.eval()
    
    # Create dynamic transform that doesn't resize to fixed size
    # NaFlex should handle variable input sizes directly
    from torchvision import transforms
    
    # Get normalization stats from model config
    data_config = resolve_model_data_config(model)
    mean = data_config.get('mean', (0.5, 0.5, 0.5))
    std = data_config.get('std', (0.5, 0.5, 0.5))
    
    # Dynamic transform: only convert to tensor and normalize, NO resizing
    dynamic_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    print(f"    Model loaded successfully")
    print(f"    Using dynamic transforms (no fixed resize)")
    
    # Test with different image sizes
    test_sizes = [
        (224, 224),   # Small
        (384, 384),   # Default
        (512, 512),   # Large
        (768, 384),   # Wide rectangle
        (384, 768),   # Tall rectangle
    ]
    
    print(f"\n>>> Testing {len(test_sizes)} different image sizes...")
    print("-" * 60)
    
    token_counts = []
    
    with torch.no_grad():
        for i, (width, height) in enumerate(test_sizes):
            # Create dummy image
            img = Image.new('RGB', (width, height), color=(100 + i*30, 150, 200))
            
            # Transform and add batch dimension
            img_tensor = dynamic_transform(img).unsqueeze(0)
            
            # Forward through model using forward_features for patch embeddings
            features = model.forward_features(img_tensor)
            
            # Get token count (should be [1, num_tokens, embedding_dim])
            if features.dim() == 2:
                features = features.unsqueeze(0)
            batch_size, num_tokens, embed_dim = features.shape
            token_counts.append(num_tokens)
            
            print(f"  Image {i+1}: {width}x{height:>4} -> "
                  f"Features: {features.shape} -> {num_tokens} tokens")
    
    print("-" * 60)
    
    # Verify variable token counts
    unique_counts = len(set(token_counts))
    print(f"\n>>> Results:")
    print(f"    Total tests: {len(test_sizes)}")
    print(f"    Unique token counts: {unique_counts}")
    print(f"    Token counts: {token_counts}")
    
    if unique_counts > 1:
        print(f"\n✅ SUCCESS: NaFlex is working! Different image sizes produce different token counts.")
        print(f"   Min tokens: {min(token_counts)}")
        print(f"   Max tokens: {max(token_counts)}")
        return True
    else:
        print(f"\n❌ FAILURE: All images produced the same token count ({token_counts[0]}).")
        print(f"   NaFlex may not be working correctly.")
        return False

if __name__ == "__main__":
    success = test_naflex_variable_tokens()
    exit(0 if success else 1)
