// nvcc -lcublas main.cu && ./a.out

// c++ it's better to include this instead of ex. stdio.h
// stdio.h gives printf, cstdio gives std::printf, avoid conflicts with other global identifiers
// prob not an issue in scripts like ours
#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <fstream>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>


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

int main(int argc, char **argv){

	CudaDeviceInfo();

	cublasHandle_t handle;
	if (cublasCreate(&handle)){
		std::cerr << "Create cublas handle error" << std::endl;
		exit(EXIT_FAILURE);
	}

	cublasDestroy(handle);

	return 0;
}
