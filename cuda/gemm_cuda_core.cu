// nvcc -lcublas gemm_cuda_core.cu && ./a.out 0

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>


// code is mostly from siboehm's blog, with annotation for understanding

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

void runCublasFP32(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C){

	cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, 
				N, A, CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);

}

// naive implementation
__global__ void sgemm_naive(int M, int K, int N, float alpha, const float* A, const float* B, float beta, float* C){

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
In the previous kernel, threads are indexed as [threadIdx.x, threadIdx.y]
Note that in a naive warp launch, threadIdx.x "changes" faster than threadIdx.y

In total, we access A[x*K + i], B[i*N + y], and C[x*N + y]
since threadIdx.x changes faster, essentially we will have a bunch of threads on different rows of A
This means every thread requires a different row of A, but the same column on B

However, in the coalesced kernel, we set it so that all the warps are in the same row of A
This way, they require the same row of A, and different columns of B
Since B is row-major, this is actually good as it means we can access different columns of B contiguously

Accessing the same value is a within-warp broadcast,
different threads accessing different values in a contiguous block is referred to as coalesced memory access
*/

// global memory coalescing
// this blocksize is size of a warp
template <const uint BLOCKSIZE>
__global__ void sgemm_global_mem_coalescing(int M, int K, int N, float alpha, const float* A, const float* B, float beta, float* C){

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

/*
Now, we will introduce tiling via shared memory. Essentially, instead of looping through K to fetch individual elements
We move by a block size. It is hard to describe how this looks like using text, animation do a better job.

This is tiling on a memory level. Using shared memory as a cache
*/
template <const uint BLOCKSIZE>
__global__ void sgemm_shared_mem_block(int M, int N, int K, float alpha,
                                       const float *A, const float *B,
                                       float beta, float *C){

    // the output block that we want to compute in this threadblock
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // the inner row & col that we're accessing in this thread
    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    // advance pointers to the starting positions
    A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
    B += cCol * BLOCKSIZE;                        // row=0, col=cCol
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE)
    {
        // Have each thread load one of the elements in A & B
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // block threads in this block until cache is fully populated
        __syncthreads();
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx)
        {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}


/*
Profiling the previous kernel shows that most of the time is being spent on fetching from shared memory.
(for individual threads, fetching shared memory into registers)
So we need to promote register reuse. This can be done by having threads compute more than 1 output.
2d Local outputs will reuse a lot of the same elements (1D also helps, but 2d is better)
increasing compute:mem_fetch ratio
*/

template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm2DBlocktiling(int M, int N, int K, float alpha, const float *A,
                       const float *B, float beta, float *C) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = threadIdx.x % (BN / TN);
  const int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in smem
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = threadIdx.x / BN;
  const uint innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}





#define M 8192
#define K 8192
#define N 8192

int CEIL_DIV(int m, int n){
    return (((m) + (n)-1) / (n));
}

bool verify_matrix(float *ref, float *mat, int size){
	for (int i = 0; i < size; i++){
		if (std::fabs(ref[i] - mat[i]) > 1e-3){
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

int main(int argc, char** argv){

    int kernel_num = 0;
    if (argc == 2){
        kernel_num = std::stoi(argv[1]);
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
        runCublasFP32(handle, M, N, K, alpha, dA, dB, beta, dC);
        break;
    case 1:
        std::cout << "running naive implementation" << std::endl;
        sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        break;
    case 2:
        std::cout << "global coalescing" << std::endl;
        sgemm_global_mem_coalescing<32><<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
        break;
    }

    // get content out of device
    cudaMemcpy(hC, dC, sizeof(float)*M*N, cudaMemcpyDeviceToHost);

    std::cout << "torch reference: \t";
    print_float(C_ref);
    std::cout << "output: \t\t";
    print_float(hC);

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
