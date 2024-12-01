// clang -O3 -std=c11 -march=native matmul_simd.c && ./a.out
// gcc -std=c11 -march=native -mavx2 -mfma matmul_simd.c && ./a.out
// around 34 flops right now, can be optimized a lot more. openblas is around 160 flops
#define _GNU_SOURCE

#include <stdio.h>
#include <stdalign.h>
#include <math.h>
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

float A[N][N];
float B[N][N];
float BT[N][N];
float C[N][N];

float verify[N][N] __attribute__((aligned(32)));

void print_float(float* values) {
    printf("float [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
           values[0], values[1], values[2], values[3],
           values[4], values[5], values[6], values[7]);
}

void print_m256(__m256 reg) {
    float* values = (float*)&reg;
    printf("[%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]\n",
           values[0], values[1], values[2], values[3],
           values[4], values[5], values[6], values[7]);
}

int main(){
    
    assert (N % 8 == 0);

    FILE *f = fopen("/tmp/gemm", "rb");
    if (f == NULL) {
        printf("Error opening file!\n");
        return -1;
    }
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wunused-result"
    fread(A, 1, sizeof(float)*N*N, f);
    fread(B, 1, sizeof(float)*N*N, f);
    fread(BT, 1, sizeof(float)*N*N, f);
    fread(verify, 1, sizeof(float)*N*N, f);
    fclose(f);
    #pragma GCC diagnostic pop

    uint64_t start = nanos();
    double flops = 2.0 * N * N * N * 1e-9;

    // note that SIMD instructions act on 8 floats at a time
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            // Initialize a SIMD register to zero
            __m256 a_vec;
            __m256 b_vec;
            __m256 c_vec = _mm256_setzero_ps();
            for (int k = 0; k < N; k += 8) {
                a_vec = _mm256_loadu_ps(&A[i][k]);
                b_vec = _mm256_loadu_ps(&BT[j][k]);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            }
            float temp[8] __attribute__((aligned(32)));
            // store contents of SIMD register into memory
            _mm256_storeu_ps(temp, c_vec);
            C[i][j] = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] + temp[7];
        }
    }

    uint64_t end = nanos();
    double gflops = (double)flops / (double)((end - start) * 1e-9);
    printf("GFLOPS: %f\n", gflops);

    for (int k = 0; k < N*N; k++) {
        if (fabsf(C[0][k]- verify[0][k]) > 1e-3) {
            printf("Verification failed\n");
            return 1;
        }
    }
    printf("Verfication successful\n");

    return 0;
}
