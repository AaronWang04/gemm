// nvcc -lcublas main_copy.cu && ./a.out

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

void print_float(float* values) {
    printf("float [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
           values[0], values[1], values[2], values[3],
           values[4], values[5], values[6], values[7]);
}



void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){

	cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &alpha, A, CUDA_R_32F, 
				N, B, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
	
	// cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, 
	// 			N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

#define M 16
#define K 16
#define N 16

int main(int argc, char **argv){

	cublasHandle_t handle;
	if (cublasCreate(&handle)){
		std::cerr << "Create cublas handle error" << std::endl;
		exit(EXIT_FAILURE);
	}

	float* A = nullptr;
	float* B = nullptr;
	float* C = nullptr;
	float* C_ref = nullptr;

	A = (float*) malloc(sizeof(float) * M * K);
	B = (float*) malloc(sizeof(float) * K * N);
	C = (float*) malloc(sizeof(float) * M * N);
	C_ref = (float*) malloc(sizeof(float) * M * N);

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

	runCublasFP32(handle, M, N, K, 1.0f, A, B, 0.0f, C);

	if(verify_matrix(C, C_ref, M*N) == false){
		std::cout << "verification failed\n";
	}

	free(A);
	free(B);
	free(C);
	free(C_ref);
	cublasDestroy(handle);

	return 0;
}
