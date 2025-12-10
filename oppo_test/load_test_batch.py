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
        # Helper to just return the tensor without unsqueeze, we will stack later
        # Actually existing preprocess_image does unsqueeze(0), so we get [1, 3, 1024, 1024]
        # stacking [1, 3, ...] will give [B, 1, 3, ...] which is wrong.
        # So we should use cat or reshape.
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

def inference_batch_pipeline(model, image_path, device, batch_size):
    """
    Simulates a full inference pipeline with batching:
    1. Load batch_size images from disk (Simulate concurrent requests)
    2. Preprocess all (CPU)
    3. Transfer to GPU (Batch)
    4. Model Inference (Batched)
    5. Post-process and Save (Simulate writing responses)
    """
    timings = {}
    try:
        t0 = time.time()
        
        # 1. Load & Preprocess Batch (CPU side)
        batch_tensors_cpu = []
        original_sizes = []
        
        if not os.path.exists(image_path):
             print(f"❌ Error: Image file not found at {os.path.abspath(image_path)}")
             return None, None

        for _ in range(batch_size):
            img = Image.open(image_path).convert("RGB")
            original_sizes.append(img.size)
            
            # Preprocess to CPU Tensor first
            transform_image = transforms.Compose([
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            inp_cpu = transform_image(img) # [3, 1024, 1024]
            batch_tensors_cpu.append(inp_cpu)
            
        # Create Batch Tensor on CPU
        input_batch_cpu = torch.stack(batch_tensors_cpu, dim=0)
        
        t1 = time.time()
        timings['load_preprocess'] = (t1 - t0) * 1000

        # 2. Transfer to GPU (ONCE)
        input_batch = input_batch_cpu.to(device)
            
        # 3. Inference (Batched)
        results = run_inference(model, input_batch, device)
        if results is None:
            return None, None
            
        t2 = time.time()
        timings['inference'] = (t2 - t1) * 1000
        
        # 4. Post-process & Save Batch
        for i in range(batch_size):
            pred = results[i].squeeze() 
            pred_pil = transforms.ToPILImage()(pred)
            pred_pil = pred_pil.resize(original_sizes[i])
            
            output_filename = f"output_batch_{i}.png"
            pred_pil.save(output_filename)
        
        t3 = time.time()
        timings['save'] = (t3 - t2) * 1000
        timings['total'] = (t3 - t0) * 1000
        
        return results, timings
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def main():
    # Configuration
    weights_path = './weights/BiRefNet-DIS-epoch_590.pth'
    image_path = "image001.jpg"
    num_iterations = 100 # Adjusted for batch size (100 * 8 = 800 images total)
    warmup_iterations = 5
    batch_size = 8
    
    # 1. Check for GPU
    print("Checking for GPU...")
    if torch.cuda.is_available():
        device = 'cuda'
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
    print(f"\nRunning {warmup_iterations} warmup iterations (Batch Size: {batch_size})...")
    for _ in range(warmup_iterations):
        inference_batch_pipeline(birefnet, image_path, device, batch_size)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # 5. Benchmark
    print(f"\nRunning {num_iterations} benchmark iterations (Batch Size: {batch_size})...")
    print(f"Total images to process: {num_iterations * batch_size}")
    
    latencies = []
    comp_times = {'load': [], 'infer': [], 'save': []}
    success_batches = 0
    fail_batches = 0
    
    start_time = time.time()
    for i in range(num_iterations):
        iter_start = time.time()
        
        result, timings = inference_batch_pipeline(birefnet, image_path, device, batch_size)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        iter_end = time.time()
        
        if result is not None:
            success_batches += 1
            latencies.append((iter_end - iter_start) * 1000) # ms
            comp_times['load'].append(timings['load_preprocess'])
            comp_times['infer'].append(timings['inference'])
            comp_times['save'].append(timings['save'])
            
            if i == 0:
                print(f"\n[Sample Breakdown] Total: {timings['total']:.2f}ms")
                print(f"  - Load & Preprocess ({batch_size} images): {timings['load_preprocess']:.2f}ms")
                print(f"  - Inference (Batch {batch_size}): {timings['inference']:.2f}ms")
                print(f"  - Save ({batch_size} images): {timings['save']:.2f}ms")
                print(f"  - Inference per Image: {timings['inference']/batch_size:.2f}ms\n")
        else:
            fail_batches += 1
        
        if (i + 1) % 10 == 0:
            print(f"Batch {i+1}/{num_iterations}...")
            
    total_time = time.time() - start_time
    
    # 7. Report
    if len(latencies) > 0:
        avg_batch_latency = np.mean(latencies)
        std_batch_latency = np.std(latencies)
        avg_load = np.mean(comp_times['load'])
        avg_infer = np.mean(comp_times['infer'])
        avg_save = np.mean(comp_times['save'])
    else:
        avg_batch_latency = 0
        std_batch_latency = 0
        avg_load = 0
        avg_infer = 0
        avg_save = 0
        
    total_successful_images = success_batches * batch_size
    fps = total_successful_images / total_time
    
    print("\n" + "="*40)
    print(f"BENCHMARK RESULTS (Batch Size: {batch_size})")
    print("="*40)
    print(f"Device: {device}")
    print(f"Image Size: {image.size}")
    print(f"Total Batches: {num_iterations}")
    print(f"Total Images: {num_iterations * batch_size}")
    print(f"Flexible Throughput: {fps:.2f} FPS")
    print(f"Batch Latency: {avg_batch_latency:.2f} ms ± {std_batch_latency:.2f} ms")
    print(f"Estimated Latency per Image: {avg_batch_latency / batch_size:.2f} ms")
    print("-" * 20)
    print("Average Component Times (per Batch):")
    print(f"  - Load & Preprocess: {avg_load:.2f} ms")
    print(f"  - Inference:       {avg_infer:.2f} ms")
    print(f"  - Save:            {avg_save:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    main()
