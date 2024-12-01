// debug purpose
// clang -O3 -std=c11 matmul_simd_equiv.c && ./a.out
// gcc -O4  -mavx matmul_simd.c && ./a.out
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

float verify[N][N];

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
    fread(A, 1, sizeof(float)*N*N, f);
    fread(B, 1, sizeof(float)*N*N, f);
    fread(BT, 1, sizeof(float)*N*N, f);
    fread(verify, 1, sizeof(float)*N*N, f);
    fclose(f);

    uint64_t start = nanos();
    double flops = 2.0 * N * N * N * 1e-9;

    // note that SIMD instructions act on 8 floats at a time
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (int k = 0; k < N; k += 8) {
                for(int l = 0; l < 8; l++){
                    acc[l] += A[i][k+l] * B[k+l][j];
                    printf("%.2f ", B[k+l][j]);
                }
                printf("\n");
            }
            exit(0);
            // store contents of SIMD register into memory
            C[i][j] = acc[0] + acc[1] + acc[2] + acc[3] + acc[4] + acc[5] + acc[6] + acc[7];
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
