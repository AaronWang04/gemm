// clang -O3 -march=native matmul_simd.c && ./a.out

// enable gnu extensions
#define _GNU_SOURCE

#include <stdio.h>
#include <stdint.h>
#include <immintrin.h> // For SIMD intrinsics
#include <time.h>
#include <assert.h>

#define N 1024
#define BLOCK_SIZE 8 // not for tiling, but for SIMD

uint64_t nanos(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000 + (uint64_t)t.tv_nsec;
}

// initialize matrices, they have to be aligned to take advantage of SIMD datatypes
// aligned to 32 bytes, which is the size of a SIMD register
float A[N*N] __attribute__ ((aligned (32)));;
float B[N*N] __attribute__ ((aligned (32)));;
float C[N*N] __attribute__ ((aligned (32)));;

// __m256 is a datatype that holds 8 single precision floats
// called a multiple accumulator register
__m256 *Am = (__m256*)A;
__m256 *Bm = (__m256*)B;
__m256 *Cm = (__m256*)C;

int main(){

    assert (N % BLOCK_SIZE == 0);

    uint64_t start = nanos();

    double flops = 2.0 * N * N * N * 1e-9;

    for(int by = 0; by < N; by += BLOCK_SIZE){
        for(int bx = 0; bx < N; bx += BLOCK_SIZE){
            __m256 tc[BLOCK_SIZE] = {};
            for(int y = 0; y < BLOCK_SIZE; y++){
                __m256 tmp;
                for(int k = 0; k < BLOCK_SIZE; k++){
                    tmp = _mm256_fmadd_ps(
                        Am[((by+y)*N+k)/8],
                        Bm[(bx*N+k)/8],
                        tmp
                    );
                }
                tc[y] = tmp;
            }

            for(int y = 0; y < BLOCK_SIZE; y++){
                Cm[(by+y)*N/8 + bx/8] = tc[y];
            }
        }
    }

    for(int x = 0; x < N; x++){
        for(int y = 0; y < N; y++){
            C[x*N + y] = Cm[x][y];
        }
    }
    
    uint64_t end = nanos();

    // printf("Time: %fs\n", (end - start)*1e-9);
    double gflops = (double)flops / (double)((end - start) * 1e-9);
    printf("%f GFLOP/S \n", gflops);
    return 0;

}
