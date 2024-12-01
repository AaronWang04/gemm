// clang -O3 -march=native matmul_simd.c && ./a.out
// around 30 flops right now, can be optimized a lot more. openblas is around 160 flops
#define _GNU_SOURCE

#include <stdio.h>
#include <stdint.h>
#include <immintrin.h> // For SIMD intrinsics
#include <time.h>
#include <assert.h>

#define N 1024

uint64_t nanos(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000 + (uint64_t)t.tv_nsec;
}

// initialize matrices, they have to be aligned to take advantage of SIMD datatypes
// aligned to 32 bytes, which is the size of a SIMD register
float A[N][N] __attribute__ ((aligned (32)));;
float B[N][N] __attribute__ ((aligned (32)));;
float C[N][N] __attribute__ ((aligned (32)));;

int main(){

    assert (N % 8 == 0);

    uint64_t start = nanos();

    double flops = 2.0 * N * N * N * 1e-9;

    // note that SIMD instructions act on 8 floats at a time
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            __m256 c_vec[8] = {};
            for (int k = 0; k < N; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i][k]);
                __m256 b_vec = _mm256_loadu_ps(&B[k][j]);
                c_vec[0] = _mm256_fmadd_ps(a_vec, b_vec, c_vec[0]);
            }

            float temp[8];
            _mm256_storeu_ps(temp, c_vec[0]);
            C[i][j] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
        }
    }

    uint64_t end = nanos();
    double gflops = (double)flops / (double)((end - start) * 1e-9);
    printf("GFLOPS: %f\n", gflops);
    return 0;
}
