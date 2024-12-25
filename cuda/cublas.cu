// nvcc -lcublas cublas.cu && ./a.out
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <fstream>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>


void print_float(float* values) {
    printf("float [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
           values[0], values[1], values[2], values[3],
           values[4], values[5], values[6], values[7]);
}


void CudaDeviceInfo(){
	int deviceId;

	cudaGetDevice(&deviceId);

	cudaDeviceProp props{};
	cudaGetDeviceProperties(&props, deviceId);

	printf("Device ID: %d\n\
    Name: %s\n\
    Compute Capability: %d.%d\n\
    memoryBusWidth: %d\n\
    maxThreadsPerBlock: %d\n\
    maxThreadsPerMultiProcessor: %d\n\
    maxRegsPerBlock: %d\n\
    maxRegsPerMultiProcessor: %d\n\
    totalGlobalMem: %zuMB\n\
    sharedMemPerBlock: %zuKB\n\
    sharedMemPerMultiprocessor: %zuKB\n\
    totalConstMem: %zuKB\n\
    multiProcessorCount: %d\n\
    Warp Size: %d\n",
		   deviceId, props.name, props.major, props.minor, props.memoryBusWidth,
		   props.maxThreadsPerBlock, props.maxThreadsPerMultiProcessor,
		   props.regsPerBlock, props.regsPerMultiprocessor,
		   props.totalGlobalMem / 1024 / 1024, props.sharedMemPerBlock / 1024,
		   props.sharedMemPerMultiprocessor / 1024, props.totalConstMem / 1024,
		   props.multiProcessorCount, props.warpSize);
}

bool verify_matrix(float *ref, float *mat, int N){
	double diff = 0.0;
	for (int i = 0; i < N; i++){
		diff = std::fabs(ref[i] - mat[i]);
		if (diff > 1e-3){
			std::cout << "failed at " << i << std::endl;
			return false;
		}
	}

	return true;
}


void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){

	// https://docs.nvidia.com/cuda/archive/12.3.2/cublas/index.html
	// CUBLAS_OP_N The non-transpose operation is selected.
	// CUBLAS_OP_T The transpose operation is selected.
	// CUBLAS_OP_C The conjugate transpose operation is selected.

	// cublasStatus_t cublasGemmEx(cublasHandle_t handle,
    //                        cublasOperation_t transa,
    //                        cublasOperation_t transb,
    //                        int m,
    //                        int n,
    //                        int k,
    //                        const void    *alpha,
    //                        const void     *A,
    //                        cudaDataType_t Atype,
    //                        int lda,
    //                        const void     *B,
    //                        cudaDataType_t Btype,
    //                        int ldb,
    //                        const void    *beta,
    //                        void           *C,
    //                        cudaDataType_t Ctype,
    //                        int ldc,
    //                        cublasComputeType_t computeType,
    //                        cublasGemmAlgo_t algo)

	// note that cublas uses column major, we must swap row-major
	// (B^T @ A^T)^T = A@B
	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, 
				N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

}

#define M 8192
#define K 8192
#define N 8192

// #define M 16
// #define K 16
// #define N 16

int main(int argc, char **argv){

	cublasHandle_t handle;
	if (cublasCreate(&handle)){
		std::cerr << "Create cublas handle error" << std::endl;
		exit(EXIT_FAILURE);
	}

	// host pointers
	float* A = nullptr;
	float* B = nullptr;
	float* C = nullptr;
	float* C_ref = nullptr;

	// device pointers
	float* dA = nullptr;
	float* dB = nullptr;
	float* dC = nullptr;

	A = (float*) malloc(sizeof(float) * M * K);
	B = (float*) malloc(sizeof(float) * K * N);
	C = (float*) calloc(M * N, sizeof(float));
	C_ref = (float*) malloc(sizeof(float) * M * N);

	cudaMalloc((void**) &dA, sizeof(float)*M*K);
	cudaMalloc((void**) &dB, sizeof(float)*K*N);
	cudaMalloc((void**) &dC, sizeof(float)*M*N);

	std::ifstream f("/tmp/gemm", std::ios::binary);
	if (!f.is_open()) {
        std::cerr << "Error opening file!\n";
        return -1;
    }

	// f.read expects a char* to read into
	// using reinterpret_cast allows you to interpret A as char* but not change its type
	f.read(reinterpret_cast<char*>(A), sizeof(float)*M*K);
	f.read(reinterpret_cast<char*>(B), sizeof(float)*K*N);
	f.read(reinterpret_cast<char*>(C_ref), sizeof(float)*M*N);
	f.close();

	// move from host to device
	cudaMemcpy(dA, A, sizeof(float)*M*K, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeof(float)*K*N, cudaMemcpyHostToDevice);
	cudaMemcpy(dC, C, sizeof(float)*M*N, cudaMemcpyHostToDevice);

	runCublasFP32(handle, M, N, K, 1.0f, dA, dB, 0.0f, dC);
	cudaDeviceSynchronize();
	cudaMemcpy(C, dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

	if(verify_matrix(C, C_ref, M*N) == false){
		std::cout << "verification failed\n";
	}

	print_float(A);
	print_float(B);
	print_float(C);
	print_float(C_ref);

	free(A);
	free(B);
	free(C);
	free(C_ref);
	cublasDestroy(handle);

	return 0;
}
