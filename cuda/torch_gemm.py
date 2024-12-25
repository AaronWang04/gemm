# python torch_gemm.py
import torch
import time

# Matrix size
N = 8192

if __name__ == "__main__":
    # theoretical number of FLOPs
    FLOP = 2*N*N*N

    A = torch.randn(N, N, device="cuda", dtype=torch.float32)
    B = torch.randn(N, N, device="cuda", dtype=torch.float32)
    C = torch.randn(N, N, device="cuda", dtype=torch.float32)

    # scalars
    alpha = 1.0
    beta = 1.0

    # warm-up
    _ = torch.addmm(beta * C, A, B, alpha=alpha)

    # measure execution time
    start = time.monotonic()
    C_result = torch.addmm(beta * C, A, B, alpha=alpha)  # GEMM operation
    end = time.monotonic()

    # calculate GFLOPS
    exec_time = end - start
    FLOPS = FLOP / exec_time
    TFLOPS = FLOPS * 1e-12
    print(f"TFLOP/S: {TFLOPS:.4f}")

    try:
        with open("/tmp/torch_gemm", "wb") as f:
            f.write(A.cpu().numpy().tobytes())
            f.write(B.cpu().numpy().tobytes())
            f.write(C.cpu().numpy().tobytes())
            f.write(C_result.cpu().numpy().tobytes())
    except Exception as e:
        print(f"Failed to save properly: {e}")
