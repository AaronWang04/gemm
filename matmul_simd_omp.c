#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <immintrin.h>
#include <time.h>

uint64_t nanos(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000 + (uint64_t)t.tv_nsec;
}

#define N 1024
#define ALIGNMENT 32

void matrix_multiply(float *A, float *B, float *C, int n) {
    int i, j, k;

    #pragma omp parallel for private(i,j,k) schedule(static)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j += 8) {
            __m256 c_vec = _mm256_load_ps(&C[i*n + j]);
            for (k = 0; k < n; k++) {
                __m256 a_vec = _mm256_broadcast_ss(&A[i*n + k]);
                __m256 b_vec = _mm256_load_ps(&B[k*n + j]);
                c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
            }
            _mm256_store_ps(&C[i*n + j], c_vec);
        }
    }
}

int main() {
    int n = N;
    int size = n * n;

    // Allocate aligned memory
    float *A = (float*) _mm_malloc(size * sizeof(float), ALIGNMENT);
    float *B = (float*) _mm_malloc(size * sizeof(float), ALIGNMENT);
    float *C = (float*) _mm_malloc(size * sizeof(float), ALIGNMENT);

    // Initialize matrices with some values
    for (int i = 0; i < size; i++) {
        A[i] = (float)(rand() % 100) / 100.0f;
        B[i] = (float)(rand() % 100) / 100.0f;
        C[i] = 0.0f;
    }

    uint64_t start = nanos();
    matrix_multiply(A, B, C, n);
    uint64_t end  = nanos();

    float flops = 2.0 * n * n * n * 1e-9;
    double gflops = (double)flops / (double)((end - start) * 1e-9);
    printf("%f GFLOP/S \n", gflops);

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);

    return 0;
}
