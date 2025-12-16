import time
import torch
import os
import sys
from PIL import Image
import numpy as np
from torchvision import transforms
import torch.multiprocessing as mp

# Add parent directory to path to allow importing from source/BiRefNet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.birefnet import BiRefNet
from utils import check_state_dict

# --- Worker Functions (Must be top-level for pickling) ---

def preprocess_worker(args):
    """
    CPU-intensive task: Load image -> Resize -> Normalize
    Returns: (input_tensor_cpu, original_size, image_path, start_time, e2e_start_time)
    """
    image_path, idx = args
    e2e_start_time = time.time()
    
    try:
        t0 = time.time()
        # 1. Load Image
        if not os.path.exists(image_path):
            return None
            
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        # 2. Transform
        transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        tensor = transform_image(image).unsqueeze(0) # [1, 3, 1024, 1024]
        
        # Return CPU tensor. Main process will move to GPU.
        preprocess_time = (time.time() - t0) * 1000
        return (tensor, original_size, idx, preprocess_time, e2e_start_time)
    
    except Exception as e:
        print(f"❌ Preprocessing failed for {image_path}: {e}")
        return None

def postprocess_worker(args):
    """
    CPU/IO-intensive task: Tensor -> PIL -> Resize -> Save
    """
    pred_tensor_cpu, original_size, idx, preprocess_time, inference_time, e2e_start_time = args
    
    try:
        t0 = time.time()
        # pred_tensor_cpu is [1, 1, 1024, 1024] or [1024, 1024]
        pred = pred_tensor_cpu.squeeze()
        
        # 1. To PIL
        pred_pil = transforms.ToPILImage()(pred)
        
        # 2. Resize back to original
        pred_pil = pred_pil.resize(original_size)
        
        # 3. Save
        output_dir = "output_multiprocessing"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"output_{idx}.png")
        pred_pil.save(output_filename)
        
        postprocess_time = (time.time() - t0) * 1000
        total_e2e_time = (time.time() - e2e_start_time) * 1000
        
        return {
            "status": "success",
            "preprocess_time": preprocess_time,
            "inference_time": inference_time,
            "postprocess_time": postprocess_time,
            "total_time": total_e2e_time
        }
    except Exception as e:
        print(f"❌ Postprocessing failed: {e}")
        return {"status": "fail"}

# --- Main Process ---

def load_model(device, weights_path):
    print(f"\nLoading BiRefNet model from local weights: {weights_path}...")
    try:
        birefnet = BiRefNet(bb_pretrained=False)
        state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
        state_dict = check_state_dict(state_dict)
        birefnet.load_state_dict(state_dict)
        birefnet.to(device)
        birefnet.eval()
        # Compile graph or warmup could happen here
        return birefnet
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

def main():
    # Settings
    mp.set_start_method('spawn', force=True) # Safe for PyTorch
    
    weights_path = './weights/BiRefNet-DIS-epoch_590.pth'
    if not os.path.exists(weights_path):
        print(f"Weights not found: {weights_path}")
        return
        
    image_path = "oppo_test/image001.jpg"
    num_iterations = 100 # Increase iterations to see throughput benefits
    num_preprocess_workers = 4
    num_postprocess_workers = 6 # IO bound, can be higher
    
    # 1. Setup Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device: {device}")
    
    # 2. Load Model (Main Process Only)
    model = load_model(device, weights_path)
    if model is None: return

    # 3. Prepare Inputs
    # We simulate processing 'num_iterations' images
    input_tasks = [(image_path, i) for i in range(num_iterations)]
    
    print(f"Starting Benchmark with:")
    print(f"- {num_iterations} Items")
    print(f"- {num_preprocess_workers} Preprocess Processes")
    print(f"- {num_postprocess_workers} Postprocess Processes")
    
    start_time = time.time()
    
    # 4. Pools
    # We use 'spawn' context implicitly via mp.set_start_method
    
    results_list = []
    
    with mp.Pool(processes=num_preprocess_workers) as pre_pool, \
         mp.Pool(processes=num_postprocess_workers) as post_pool:
        
        # Producer: Preprocess
        # imap returns an iterator that yields results as they finish
        # This keeps the pipeline fed without loading everything into memory at once
        pre_results_iterator = pre_pool.imap(preprocess_worker, input_tasks)
        
        post_async_results = []
        
        print("\nProcessing streaming...")
        
        processed_count = 0
        
        # Main Inference Loop
        for pre_result in pre_results_iterator:
            if pre_result is None:
                continue
                
            tensor_cpu, original_size, idx, prep_time, e2e_start = pre_result
            
            # --- Inference (GPU) ---
            t_inf_start = time.time()
            with torch.no_grad():
                # Move to GPU
                input_gpu = tensor_cpu.to(device, non_blocking=True)
                
                # AutoCast
                if device == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        preds = model(input_gpu)[-1].sigmoid()
                else:
                    preds = model(input_gpu)[-1].sigmoid()
                
                # Move back to CPU for postprocessing
                preds_cpu = preds.detach().cpu()
            
            if device == 'cuda':
                torch.cuda.synchronize()
            t_inf_end = time.time()
            inf_time = (t_inf_end - t_inf_start) * 1000
            
            # --- Send to Postprocessor ---
            # We use apply_async to fire and forget (mostly), collecting result objects to check later
            post_args = (preds_cpu, original_size, idx, prep_time, inf_time, e2e_start)
            res = post_pool.apply_async(postprocess_worker, args=(post_args,))
            post_async_results.append(res)
            
            processed_count += 1
            if processed_count % 10 == 0:
                print(f"Inference Progress: {processed_count}/{num_iterations}")

        # Collect results
        print("Inference finished. Waiting for post-processing...")
        for res in post_async_results:
            results_list.append(res.get())
            
    total_time = time.time() - start_time
    
    # 5. Analyze
    success_count = sum(1 for r in results_list if r['status'] == 'success')
    fps = success_count / total_time
    
    avg_prep = np.mean([r['preprocess_time'] for r in results_list if r['status'] == 'success'])
    avg_inf = np.mean([r['inference_time'] for r in results_list if r['status'] == 'success'])
    avg_post = np.mean([r['postprocess_time'] for r in results_list if r['status'] == 'success'])
    avg_total = np.mean([r['total_time'] for r in results_list if r['status'] == 'success'])
    
    print("\n" + "="*40)
    print("MULTIPROCESSING BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Time: {total_time:.4f}s")
    print(f"Throughput: {fps:.2f} FPS")
    if fps > 0:
        print(f"Avg Time per Image: {1000/fps:.2f} ms")
    print("-" * 20)
    print("Average Component Latencies (Per Sample):")
    print(f"  Preprocess (CPU):  {avg_prep:.2f} ms")
    print(f"  Inference (GPU):   {avg_inf:.2f} ms")
    print(f"  Postprocess (CPU): {avg_post:.2f} ms")
    print(f"  Total E2E:         {avg_total:.2f} ms")
    print("="*40)

if __name__ == "__main__":
    main()
