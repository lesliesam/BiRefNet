import time
import torch
import os
import sys
from PIL import Image
import numpy as np
from torchvision import transforms
from io import BytesIO

# Add parent directory to path to allow importing from source/BiRefNet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.birefnet import BiRefNet
from utils import check_state_dict

def load_model(device, weights_path):
    print(f"\nLoading BiRefNet model from local weights: {weights_path}...")
    
    if not os.path.exists(weights_path):
        print(f"❌ Error: Weights file not found at {os.path.abspath(weights_path)}")
        print("Please make sure you are running this script from the 'source/BiRefNet' directory")
        print("and that the weights are in 'source/BiRefNet/weights/BiRefNet-DIS-epoch_590.pth'")
        return None

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
        return birefnet
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return None

def preprocess_image(image, device):
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    try:
        input_images = transform_image(image).unsqueeze(0).to(device)
        return input_images
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return None

def run_inference(model, input_images, device):
    try:
        with torch.no_grad():
            # Use mixed precision if on GPU for speed (optional but good for L4)
            if device == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    preds = model(input_images)[-1].sigmoid()
            else:
                preds = model(input_images)[-1].sigmoid()
            
            preds = preds.cpu()
        return preds
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        return None

def inference_pipeline(model, image_path, device):
    """
    Simulates a full inference pipeline:
    1. Load image from disk (Simulate reading file)
    2. Preprocess
    3. Model Inference
    4. Post-process and Save (Simulate writing response)
    """
    try:
        # 1. Load from disk
        if not os.path.exists(image_path):
            print(f"❌ Error: Image file not found at {os.path.abspath(image_path)}")
            return None
        img = Image.open(image_path).convert("RGB")
        
        # 2. Preprocess
        inp = preprocess_image(img, device)
        if inp is None:
            return None
            
        # 3. Inference
        result = run_inference(model, inp, device)
        
        # 4. Post-process & Save
        if result is None:
            return None
            
        pred = result[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        pred_pil = pred_pil.resize(img.size)
        
        output_filename = "output_loadtest.png"
        pred_pil.save(output_filename)
        
        return result
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return None

def main():
    # Configuration
    weights_path = './weights/BiRefNet-DIS-epoch_590.pth'
    image_path = "oppo_test/image001.jpg"
    num_iterations = 20
    warmup_iterations = 10
    
    # 1. Check for GPU
    print("Checking for GPU...")
    if torch.cuda.is_available():
        device = 'cuda'
        print("CUDA Version:", torch.version.cuda)
        print(f"✅ CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️ CUDA not available. Using CPU (this will be slow).")

    # 2. Load Model
    birefnet = load_model(device, weights_path)
    if birefnet is None:
        return

    # 3. Verify Image Exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found at {os.path.abspath(image_path)}")
        return
    image = Image.open(image_path)
    print(f"✅ Image found at {os.path.abspath(image_path)}. Size: {image.size}")

    # 4. Warmup
    print(f"\nRunning {warmup_iterations} warmup iterations (full pipeline)...")
    for _ in range(warmup_iterations):
        inference_pipeline(birefnet, image_path, device)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 5. Benchmark
    print(f"Running {num_iterations} benchmark iterations (full pipeline)...")
    latencies = []
    success_count = 0
    fail_count = 0
    
    start_time = time.time()
    for i in range(num_iterations):
        iter_start = time.time()
        
        # Simulate request handling: Load Disk -> Preprocess -> Inference
        result = inference_pipeline(birefnet, image_path, device)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        iter_end = time.time()
        
        if result is not None:
            success_count += 1
            latencies.append((iter_end - iter_start) * 1000) # ms
        else:
            fail_count += 1
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{num_iterations}...")
            
    total_time = time.time() - start_time
    
    # 7. Report
    if len(latencies) > 0:
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
    else:
        avg_latency = 0
        std_latency = 0
        
    fps = success_count / total_time
    
    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print("="*40)
    print(f"Device: {device}")
    print(f"Image Size: {image.size}")
    print(f"Total Iterations: {num_iterations}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Average Latency (Success): {avg_latency:.2f} ms ± {std_latency:.2f} ms")
    print(f"Throughput (Success): {fps:.2f} FPS")
    print("="*40)

if __name__ == "__main__":
    main()
