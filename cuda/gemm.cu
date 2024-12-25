#include <cuda_runtime.h>
#include <cstdio.h>
#include <cstdlib.h>

// annotated from siboehm's blog

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

// naive implementation
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

// global memory coalescing
template <const uint BLOCKSIZE>
__global__ void gemm_global_coalescing(int M, int K, int N, float alpha, const float* A, const float* B, float beta, float* C)){

    const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if(x < M && y < N){
        float tmp = 0.0;
        for(int i = 0; i < K; i++){
            tmp += A[x*K + i] * B[i*N + y];
        }
        C[x*N + y] = alpha*tmp + beta*C[x*N + y];
    }

}

int main(int argc, char** argv){

    if argc == 2{
        int kernel_num = std::stoi(argv[1]);
    }
    else{
        
    }

}
