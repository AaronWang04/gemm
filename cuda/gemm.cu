// nvcc -lcublas gemm.cu && ./a.out 0

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>


// code is from siboehm's blog, with annotation for understanding

// matrix M*K @ K*N = M*N
// matmul is C = A@B
// gemm is C = α*(A@B)+β*C, * for scalar multiplication

__global__ void matmul(int M, int K, int N, const float* A, const float* B, float* C){

    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    // if statement is needed for tile quantization
    if(x < M && y < N){
        float tmp = 0.0;
        for(int i = 0; i < K; i++){
            tmp += A[x*K + i] * B[i*N + y];

        }

        C[x*N + y] = tmp;

    }

}


/*
Matrixes are stored in row-majored
When looping, A is consecutive in memory, B is not
*/

// naive implementation
__global__ void gemm_naive(int M, int K, int N, float alpha, const float* A, const float* B, float beta, float* C){

    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x < M && y < N){
        float tmp = 0.0;
        for(int i = 0; i < K; i++){
            // load from A
            // load from B
            tmp += A[x*K + i] * B[i*N + y];
        }

        // load from C
        // store to C
        C[x*N + y] = alpha*tmp + beta*C[x*N + y];

    }

}

/*
The problem with the previous implementation comes from 

*/
// global memory coalescing
// this blocksize is size of a warp
template <const uint BLOCKSIZE>
__global__ void gemm_global_mem_coalescing(int M, int K, int N, float alpha, const float* A, const float* B, float beta, float* C){

    // this ensures that 
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

void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){

	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, 
				N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

}


#define M 8192
#define K 8192
#define N 8192
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


bool verify_matrix(float *ref, float *mat, int size){
	for (int i = 0; i < size; i++){
		if (std::fabs(ref[i] - mat[i]) > 1e-3){
			std::cout << "failed at " << i << std::endl;
			return false;
		}
	}

	return true;
}

int main(int argc, char** argv){

    int kernel_num = 0;
    if (argc == 2){
        int kernel_num = std::stoi(argv[1]);
    }

    // when initializing multiple pointers, you have to do this disgusting *varname convention blegh
    float *hA = nullptr, *hB = nullptr, *hC = nullptr, *C_ref = nullptr;
    float *dA = nullptr, *dB = nullptr, *dC = nullptr;

    hA = (float*)malloc(sizeof(float) * M * K);
	hB = (float*)malloc(sizeof(float) * K * N);
	hC = (float*)malloc(sizeof(float) * M * N);
	C_ref = (float*)malloc(sizeof(float) * M * N);

    cudaMalloc((void**) &dA, sizeof(float)*M*K);
	cudaMalloc((void**) &dB, sizeof(float)*K*N);
	cudaMalloc((void**) &dC, sizeof(float)*M*N);

    std::ifstream f("/tmp/torch_gemm", std::ios::binary);
	if (!f.is_open()) {
        std::cerr << "Error opening file!\n";
        return -1;
    }

    f.read(reinterpret_cast<char*>(hA), sizeof(float)*M*K);
	f.read(reinterpret_cast<char*>(hB), sizeof(float)*K*N);
	f.read(reinterpret_cast<char*>(hC), sizeof(float)*K*N);
	f.read(reinterpret_cast<char*>(C_ref), sizeof(float)*M*N);
	f.close();

    cudaMemcpy(dA, hA, sizeof(float)*M*K, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, sizeof(float)*K*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dC, hC, sizeof(float)*M*N, cudaMemcpyHostToDevice);

    float alpha = 1.0f;
    float beta = 1.0f;


    cublasHandle_t handle;
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 blockDim(32, 32);
    switch(kernel_num){
    case 0:
        std::cout << "running cublas" << std::endl;
        if (cublasCreate(&handle)){
            std::cerr << "Create cublas handle error" << std::endl;
            exit(EXIT_FAILURE);
        }
        break;
    case 1:
        std::cout << "running naive implementation" << std::endl;
        gemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        break;
    case 2:
        std::cout << "global coalescing" << std::endl;
        gemm_global_mem_coalescing<32><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        break;
    }

    // get content out of device
    cudaMemcpy(hC, dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

    if(verify_matrix(hC, C_ref, M*N) == false){
		std::cout << "verification failed\n";
	}

    free(hA);
	free(hB);
	free(hC);
	free(C_ref);
    cudaFree(dB);
    cudaFree(dA);
    cudaFree(dC);
	cublasDestroy(handle);

    return 0;
}
