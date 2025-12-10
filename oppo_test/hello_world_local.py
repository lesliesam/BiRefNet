import torch
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import os
import sys

# Add parent directory to path to allow importing from source/BiRefNet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.birefnet import BiRefNet
from utils import check_state_dict

def main():
    # 1. Check for GPU
    print("Checking for GPU...")
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"✅ CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️ CUDA not available. Using CPU (this will be slow).")

    # 2. Load Model
    weights_path = './weights/BiRefNet-DIS-epoch_590.pth'
    print(f"\nLoading BiRefNet model from local weights: {weights_path}...")
    
    if not os.path.exists(weights_path):
        print(f"❌ Error: Weights file not found at {os.path.abspath(weights_path)}")
        print("Please make sure you are running this script from the 'source/BiRefNet' directory")
        print("and that the weights are in 'source/BiRefNet/weights/BiRefNet-DIS-epoch_590.pth'")
        return

    try:
        # Initialize model structure
        birefnet = BiRefNet(bb_pretrained=False)
        
        # Load weights
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        state_dict = check_state_dict(state_dict)
        birefnet.load_state_dict(state_dict)
        
        birefnet.to(device)
        birefnet.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Load Sample Image
    image_path = "image001.jpg"
    print(f"\nLoading sample image from {image_path}...")
    try:
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found at {os.path.abspath(image_path)}")
            return
        image = Image.open(image_path).convert("RGB")
        print(f"✅ Image loaded. Size: {image.size}")
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return

    # 4. Preprocess
    print("\nPreprocessing image...")
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        input_images = transform_image(image).unsqueeze(0).to(device)
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return

    # 5. Inference
    print("Running inference...")
    try:
        with torch.no_grad():
            # Use mixed precision if on GPU for speed (optional but good for L4)
            if device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    preds = birefnet(input_images)[-1].sigmoid()
            else:
                preds = birefnet(input_images)[-1].sigmoid()
            
            preds = preds.cpu()
        
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        pred_pil = pred_pil.resize(image.size)
        print("✅ Inference complete.")
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return
    
    # 6. Save Result
    output_filename = "hello_world_local_result.png"
    try:
        pred_pil.save(output_filename)
        print(f"\n🎉 Success! Result saved to: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"❌ Failed to save result: {e}")

if __name__ == "__main__":
    main()
