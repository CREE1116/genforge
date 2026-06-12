"""
Verify and benchmark GPU (MPS) support in OQBoostResearch.
"""
from __future__ import annotations

import os
import time
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.oqboost_research import OQBoostResearch


def main():
    print("==================================================")
    print("GPU (MPS) Backend Verification & Benchmark")
    print("==================================================")
    
    # 1. Check availability
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"MPS Available: {torch.backends.mps.is_available()}")
    
    # Generate mock classification dataset (5,000 samples, 30 features)
    print("\nGenerating benchmark dataset...")
    X, y = make_classification(
        n_samples=5000,
        n_features=30,
        n_informative=15,
        random_state=42
    )
    X = StandardScaler().fit_transform(X).astype(np.float32)
    
    # 2. Benchmark CPU
    print("\n[Benchmarking CPU]")
    t0 = time.time()
    clf_cpu = OQBoostResearch(
        n_estimators=30,
        learning_rate=0.1,
        max_depth=4,
        use_wls=True,
        inherit_mode='orth',
        orth_strategy='random',
        random_state=42,
        device='cpu'
    )
    clf_cpu.fit(X, y)
    cpu_time = time.time() - t0
    cpu_loss = clf_cpu.train_losses_[-1]
    print(f"  CPU Training Time: {cpu_time:.2f}s | Final Train Loss: {cpu_loss:.4f}")
    
    # 3. Benchmark GPU (MPS)
    target_device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    if target_device == 'cpu':
        print("\nGPU (MPS) not available. Skipping GPU benchmark.")
        return
        
    print(f"\n[Benchmarking GPU ({target_device.upper()})]")
    t0 = time.time()
    clf_gpu = OQBoostResearch(
        n_estimators=30,
        learning_rate=0.1,
        max_depth=4,
        use_wls=True,
        inherit_mode='orth',
        orth_strategy='random',
        random_state=42,
        device=target_device
    )
    clf_gpu.fit(X, y)
    gpu_time = time.time() - t0
    gpu_loss = clf_gpu.train_losses_[-1]
    print(f"  {target_device.upper()} Training Time: {gpu_time:.2f}s | Final Train Loss: {gpu_loss:.4f}")
    
    # 4. Verification
    speedup = cpu_time / max(gpu_time, 1e-6)
    print(f"\nSpeedup: {speedup:.2f}x")
    
    # Verify that predictions/losses are close
    diff = abs(cpu_loss - gpu_loss)
    print(f"Loss Difference between CPU and GPU: {diff:.6e}")
    if diff < 1e-4:
        print("SUCCESS: CPU and GPU outputs match closely!")
    else:
        print("WARNING: Outputs differ, please check numerical precision.")


if __name__ == "__main__":
    main()
