#include <cuda_runtime.h>
// old version of cublas api, less optimized but supported on older platforms
// #include <cublas.cu>
#include <cublas_v2.cu>


#define M 8192
#define K 8192
#define N 8192


int main(int argc, char* argv[]){

    cudaError_t cudaStat;   // cudamalloc status
    cublasStatus_t stat;    // cublas function status
    cublasHandle_t handle;  // cublas context

    a = (float*)malloc(M*K*sizeof(float));
    b = (float*)malloc(K*N*sizeof(float));
    c = (float*)malloc(M*N*sizeof(float));



}
