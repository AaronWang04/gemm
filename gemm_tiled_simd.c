// clang -O3 -march=native -ffast-math gemm_tiled_simd.c && ./a.out

#define _GNU_SOURCE

#include <stdio.h>
#include <stdalign.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h> // For SIMD intrinsics
#include <time.h>
#include <assert.h>

#define N 1024
#define BLOCK_SIZE 8
#define REGISTER_SIZE 8

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

float A[N][N];
float AT[N][N];
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
                AT[i + l][j] = A[j][i + l];
            }
        }
    }
    __m256 a_vec;
    __m256 b_vec;
    __m256 c_vec;
    for (int bx = 0; bx < N; bx += BLOCK_SIZE){
        for (int by = 0; by < N; by += BLOCK_SIZE){

            __m256 vec_arr[BLOCK_SIZE];
            for(int i = 0; i < BLOCK_SIZE; i++){
                vec_arr[i] = _mm256_setzero_ps();
            }

            for (int k = 0; k < N; k++){
                b_vec = _mm256_loadu_ps(&B[k][by]);
                for (int x = bx; x < bx + BLOCK_SIZE; x++){
                    a_vec = _mm256_broadcast_ss(&AT[k][x]);
                    vec_arr[x-bx] = _mm256_fmadd_ps(a_vec, b_vec, vec_arr[x-bx]);
                }
            }

            float temp[8] __attribute__((aligned(32)));
            for(int i = 0; i < BLOCK_SIZE; i++){
                _mm256_storeu_ps(temp, vec_arr[i]);
                for(int j = 0; j < BLOCK_SIZE; j++){
                    C[bx+i][by+j] = temp[j];
                }
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
