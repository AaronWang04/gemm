#pragma once
#include <cuda_runtime.h>
#include <cstdio.h>
#include <cstdlib.h>

// matrix M*K @ K*N = M*N

// matmul is C = A@B
// gemm is C = α*(A@B)+β*C, * for scalar multiplication

__global__ void matmul(int M, int K, int N, const float* A, const float* B, float* C){

    const uint x = blockIdx.x * blockDim.x + threadIdx.x
    const uint y = blockIdx.y * blockDim.y + threadIdx.y

    // if statement is needed for tile quantization
    if(x < M && y < N){
        float tmp = 0.0;
        for(int i = 0; i < K; i++){
            tmp += A[x*K + i] * B[i*N + y];

        }

        C[x*N + y] = tmp;

    }

}

__global__ void gemm(int M, int K, int N, float alpha, const float* A, const float* B, float beta, float* C)){

    const uint x = blockIdx.x * blockDim.x + threadIdx.x
    const uint y = blockIdx.y * blockDim.y + threadIdx.y

    if(x < M && y < N){
        float tmp = 0.0;
        for(int i = 0; i < K; i++){
            tmp += A[x*K + i] * B[i*N + y];
        }

        C[x*N + y] = alpha*tmp + beta*C[x*N + y];

    }

}
