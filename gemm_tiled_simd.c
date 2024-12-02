// clang -O3 -std=c11 -march=native matmul_tiled_simd.c && ./a.out

#define _GNU_SOURCE

#include <stdio.h>
#include <stdalign.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h> // For SIMD intrinsics
#include <time.h>
#include <assert.h>

#define N 32
#define BLOCK_SIZE 8

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
    fread(verify, 1, sizeof(float)*N*N, f);
    fclose(f);
    #pragma GCC diagnostic pop

    double flops = 2.0 * N * N * N * 1e-9;
    uint64_t start = nanos();

    // swizzling for transpose
    for (int i = 0; i < N; i+=8) {
        for (int j = 0; j < N; j++) {
            for (int l = 0; l < 8; l++) {
                BT[i + l][j] = B[j][i + l];
            }
        }
    }

    for(int by = 0; by < N; by += BLOCK_SIZE){
        for(int bx = 0; bx < N; bx += BLOCK_SIZE){
            
            for (int k = 0; k < N; k++){
                
            }

        }
    }

    uint64_t end = nanos();
    double gflops = (double)flops / (double)((end - start) * 1e-9);
    printf("GFLOP/s: %f\n", gflops);

    for (int k = 0; k < N*N; k++) {
        if (fabsf(C[0][k]- verify[0][k]) > 1e-3) {
            printf("Verification failed\n");
            return 1;
        }
    }
    printf("Verfication successful\n");

    return 0;
}
