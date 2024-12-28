/*
Tensor cores are basically asics for matmuls, a lot faster than rawdogging math with cuda cores
*/

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

#define M 8192
#define K 8192
#define N 8192

bool verify_matrix(float *ref, float *mat, int size){
	for (int i = 0; i < size; i++){
		if (std::fabs(ref[i] - mat[i]) > 1e-3){
			std::cout << "failed at " << i << std::endl;
			return false;
		}
	}

	return true;
}

int main(){

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

    cudaMemcpy(hC, dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

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
