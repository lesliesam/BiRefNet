import cv2
import numpy as np
import os
import sys
import cupy as cp

import cvcuda
import torch

package_root = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'python')
sys.path.insert(0, package_root)

# from aiboost.vision.clib.opencv import resize

import time

def get_cvcuda_tensor(input_data, device):
    frame_nhwc = None
    if isinstance(input_data, torch.Tensor):
        frame_nhwc = cvcuda.as_tensor(input_data, "NHWC")
    elif isinstance(input_data, np.ndarray):
        if input_data.ndim == 2:
            frame_nhwc = cvcuda.as_tensor(
                torch.as_tensor(input_data).unsqueeze(0).unsqueeze(-1).contiguous().to(device=device, non_blocking=True),
                "NHWC",
            )
        else:
            frame_nhwc = cvcuda.as_tensor(
                torch.as_tensor(input_data).unsqueeze(0).contiguous().to(device=device, non_blocking=True),
                "NHWC",
            )
    elif isinstance(input_data, cp.ndarray):
        if input_data.ndim == 2:
            frame_nhwc = cvcuda.as_tensor(
                torch.as_tensor(input_data).unsqueeze(0).unsqueeze(-1).contiguous().to(device=device, non_blocking=True),
                "NHWC",
            )
        else:
            frame_nhwc = cvcuda.as_tensor(
                torch.as_tensor(input_data).unsqueeze(0).contiguous().to(device=device, non_blocking=True),
                "NHWC",
            )
    return frame_nhwc


def resize(src, dsize, dst=None, fx=None, fy=None, interpolation=cv2.INTER_AREA):
    device = "cuda:0"

    src_t = get_cvcuda_tensor(src, device)

    cvcuda_interpolation = cvcuda.Interp.LINEAR if interpolation == cv2.INTER_LINEAR else cvcuda.Interp.AREA
    result = cvcuda.resize(
        src_t,
        (
            src_t.shape[0],
            dsize[1],
            dsize[0],
            src_t.shape[3],
        ),
        cvcuda_interpolation,
    )

    cp_array_result = cp.asarray(result.cuda())
    res = cp_array_result[0]

    return res



if __name__ == "__main__":
    # 定义不同的测试尺寸
    test_sizes = [
        (640, 480, 1280, 960),      # VGA -> 2x放大
        (1280, 720, 1920, 1080),    # HD -> FHD
        (1920, 1080, 3840, 2160),   # FHD -> 4K
        (1920, 1080, 1920, 1080),   # FHD -> FHD (无缩放)
        (3840, 2160, 1920, 1080),   # 4K -> FHD (下采样)
        (3840, 2160, 7680, 4320),   # 4K -> 8K (2x放大)
        
        (4096, 3072, 8192, 6144),   # 原始尺寸 -> 2x放大
        (8192, 6144, 4096, 3072),   # 8K -> 4K (下采样)
        (2048, 1536, 4096, 3072),   # 中等尺寸 -> 2x放大
        (1024, 768, 4096, 3072),    # 小尺寸 -> 4x放大
        
        (512, 384, 2048, 1536),     # 很小尺寸 -> 4x放大
        (256, 192, 1024, 768),      # 极小尺寸 -> 4x放大
        (128, 96, 512, 384),        # 超小尺寸 -> 4x放大
        (64, 48, 256, 192),         # 微型尺寸 -> 4x放大
        
        (1920, 1080, 960, 540),     # FHD -> 0.5x缩小
        (1920, 1080, 2880, 1620),   # FHD -> 1.5x放大
        (1920, 1080, 5760, 3240),   # FHD -> 3x放大
        (1024, 768, 8192, 6144),    # 小尺寸 -> 8x放大
        
        (1920, 1440, 3840, 2880),   # 4:3 -> 2x放大
        (2560, 1440, 5120, 2880),   # 2K -> 4K
        (1280, 1024, 2560, 2048),   # SXGA -> 2x放大

        (1024, 1024, 2048, 2048),   # 1K方形 -> 2K方形
        (2048, 2048, 4096, 4096),   # 2K方形 -> 4K方形
        (512, 512, 4096, 4096),     # 512方形 -> 4K方形

        # (4096, 3072, 16384, 12288), # 4K -> 16K
        (16384, 12288, 4096, 3072), # 16K -> 4K
        # (2048, 1536, 16384, 12288), # 2K -> 16K
    ]
    
    num_iterations = 100
    
    img_path = './tests/input/X8_IMG20241225160045_02.jpg'
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        sys.exit(1)
    
    original_img = cv2.imread(img_path)
    if original_img is None:
        print(f"Error: Failed to load image from {img_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("GPU vs CPU Performance Benchmark - Resize Operation")
    print("=" * 80)
    print(f"Original image size: {original_img.shape[1]}x{original_img.shape[0]}")
    print(f"Number of iterations per test: {num_iterations}")
    print("=" * 80)
    print()
    
    # Warmup
    print("Warming up...")
    warmup_img = cv2.resize(original_img, (1920, 1080))
    warmup_cupy = cp.asarray(warmup_img)
    for i in range(5):
        _ = cv2.resize(warmup_img, (1920, 1080), interpolation=cv2.INTER_LINEAR)
        _ = resize(warmup_cupy, (1920, 1080), interpolation=cv2.INTER_LINEAR)
    print("Warmup completed.\n")
    
    # 对每个尺寸进行测试
    results = []
    
    for idx, (in_w, in_h, out_w, out_h) in enumerate(test_sizes, 1):
        print(f"Test {idx}/{len(test_sizes)}: Input {in_w}x{in_h} -> Output {out_w}x{out_h}")
        print("-" * 80)
        
        input_img = cv2.resize(original_img, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        input_cupy = cp.asarray(input_img)
        
        cpu_times = []
        for i in range(num_iterations):
            start_time = time.time()
            result_cpu = cv2.resize(input_img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            cpu_times.append((time.time() - start_time) * 1000)
        
        cpu_avg = np.mean(cpu_times)
        cpu_min = np.min(cpu_times)
        cpu_max = np.max(cpu_times)
        cpu_std = np.std(cpu_times)
        
        gpu_times = []
        for i in range(num_iterations):
            start_time = time.time()
            result_gpu = resize(input_cupy, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            cp.cuda.Stream.null.synchronize()
            gpu_times.append((time.time() - start_time) * 1000)
            result_gpu = result_gpu.get() #d2h
        
        gpu_avg = np.mean(gpu_times)
        gpu_min = np.min(gpu_times)
        gpu_max = np.max(gpu_times)
        gpu_std = np.std(gpu_times)
        
        # 加速比
        speedup = cpu_avg / gpu_avg if gpu_avg > 0 else 0
        
        results.append({
            'input_size': (in_w, in_h),
            'output_size': (out_w, out_h),
            'cpu_avg': cpu_avg,
            'cpu_min': cpu_min,
            'cpu_max': cpu_max,
            'cpu_std': cpu_std,
            'gpu_avg': gpu_avg,
            'gpu_min': gpu_min,
            'gpu_max': gpu_max,
            'gpu_std': gpu_std,
            'speedup': speedup
        })
        
        print(f"CPU (cv2.resize):")
        print(f"  Average: {cpu_avg:.3f} ms")
        print(f"  Min:     {cpu_min:.3f} ms")
        print(f"  Max:     {cpu_max:.3f} ms")
        print(f"  Std:     {cpu_std:.3f} ms")
        print()
        print(f"GPU (cvcuda resize):")
        print(f"  Average: {gpu_avg:.3f} ms")
        print(f"  Min:     {gpu_min:.3f} ms")
        print(f"  Max:     {gpu_max:.3f} ms")
        print(f"  Std:     {gpu_std:.3f} ms")
        print()
        print(f"Speedup: {speedup:.2f}x {'(GPU faster)' if speedup > 1 else '(CPU faster)'}")
        print("=" * 80)
        print()
        
        if idx == len(test_sizes):
            output_path = f'output_{out_w}x{out_h}.jpg'
            cv2.imwrite(output_path, result_gpu)
            print(f"Output image saved to: {output_path}\n")
    
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'Input Size':<15} {'Output Size':<15} {'CPU Avg (ms)':<15} {'GPU Avg (ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    for r in results:
        in_size = f"{r['input_size'][0]}x{r['input_size'][1]}"
        out_size = f"{r['output_size'][0]}x{r['output_size'][1]}"
        print(f"{in_size:<15} {out_size:<15} {r['cpu_avg']:<15.3f} {r['gpu_avg']:<15.3f} {r['speedup']:<10.2f}x")
    print("=" * 80)