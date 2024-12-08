// clang -O3 -march=native -ffast-math gemm_tiled_simd_2.c && ./a.out
// same speed as regular gemm_simd, not sure how to achieve cache coherency, need to mess around?
#define _GNU_SOURCE

#include <stdio.h>
#include <stdalign.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h> // For SIMD intrinsics
#include <time.h>
#include <assert.h>

#define N 1024
#define BLOCK_Y 4
#define BLOCK_X 2

uint64_t nanos(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000 + (uint64_t)t.tv_nsec;
}


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

// TODO: check if alignment helps with cache hits
float A[N*N] __attribute__((aligned(32)));
float AT[N*N] __attribute__((aligned(32)));
float B[N*N] __attribute__((aligned(32)));
float BT[N*N] __attribute__((aligned(32)));
float C[N*N] __attribute__((aligned(32)));
float verify[N*N] __attribute__((aligned(32)));

__m256 *Am = (__m256*)A;
__m256 *Bm = (__m256*)BT;
__m256 *Cm = (__m256*)C;

void matmul(){

    for(int by = 0; by < N; by += BLOCK_Y){
        for(int bx = 0; bx < N; bx += BLOCK_X){

            // compute
            __m256 tc[BLOCK_Y][BLOCK_X] = {};
            for(int k = 0; k < N; k+= 8){
                for(int y = 0; y < BLOCK_Y; y++){
                    __m256 ty = Am[((by+y)*N + k)/ 8];
                    for(int x = 0; x < BLOCK_X; x++){
                        tc[y][x] = _mm256_fmadd_ps(ty, Bm[((bx+x)*N + k)/ 8], tc[y][x]);
                    }
                }
            }

            // store
            for(int y = 0; y < BLOCK_Y; y++){
                for(int x = 0; x < BLOCK_X; x++){
                    float ftmp = 0.0;
                    for(int i = 0; i < 8; i++) ftmp += tc[y][x][i];
                    C[(by+y)*N + bx+x] = ftmp;
                }
            }


        }
    }

}

int main(){

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

    for (int i = 0; i < N; i+=8) {
        for (int j = 0; j < N; j++) {
            for (int l = 0; l < 8; l++) {
                BT[(i+l)*N+j] = B[j*N + (i+l)];
                AT[(i+l)*N+j] = A[j*N + (i+l)];
            }
        }
    }

    matmul();

    uint64_t end = nanos();
    double gflops = (double)flops / (double)((end - start) * 1e-9);
    printf("GFLOP/s: %f\n", gflops);

    for (int k = 0; k < N*N; k++) {
        if (fabsf(C[k]- verify[k]) > 1e-3) {
            printf("Verification failed\n");
            return 1;
        }
    }
    printf("Verfication successful\n");

}
