import torch
from transformers import AutoModelForImageSegmentation
from torchvision import transforms
from PIL import Image
import requests
from io import BytesIO
import os

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
    print("\nLoading BiRefNet model from Hugging Face (zhengpeng7/BiRefNet)...")
    try:
        birefnet = AutoModelForImageSegmentation.from_pretrained('zhengpeng7/BiRefNet', trust_remote_code=True)
        birefnet.to(device)
        birefnet.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 3. Load Sample Image
    url = "https://images.unsplash.com/photo-1543466835-00a7907e9de1?q=80&w=1000&auto=format&fit=crop" # Cute dog
    print(f"\nDownloading sample image from {url}...")
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        print(f"✅ Image loaded. Size: {image.size}")
    except Exception as e:
        print(f"❌ Failed to download/load image: {e}")
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
    output_filename = "hello_world_result.png"
    try:
        pred_pil.save(output_filename)
        print(f"\n🎉 Success! Result saved to: {os.path.abspath(output_filename)}")
    except Exception as e:
        print(f"❌ Failed to save result: {e}")

if __name__ == "__main__":
    main()
