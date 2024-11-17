import numpy as np
import time

N = 2048

A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)
FLOP = 2*N**3

for _ in range(10):
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)
    start = time.perf_counter()
    C = A @ B
    end = time.perf_counter()
    exec_time = end - start
    FLOPS = FLOP/exec_time
    GFLOPS = FLOPS/1e9
    print(f"GFLOPS: {GFLOPS}")
