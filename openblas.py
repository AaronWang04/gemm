import os
# use only 1 thread
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# make sure the import is done after the environment variable is set
import numpy as np
import time

N = 1024
if __name__ == "__main__":
    FLOP = 2*N*N*N
    A = np.random.randn(N, N).astype(np.float32)
    B = np.random.randn(N, N).astype(np.float32)

    # Warm-up
    for _ in range(5):
        _ = A @ B

    for _ in range(10):
        start = time.monotonic()
        C = A @ B
        end = time.monotonic()
        exec_time = end - start
        FLOPS = FLOP/exec_time
        GFLOPS = FLOPS*1e-9
        print(f"GFLOP/S: {GFLOPS}")
