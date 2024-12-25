// clang -O3 avx/matmul_basic.c && ./a.out
// ~1 GFLOP/s
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

#define N 1024

float A[N][N];
float B[N][N];
float C[N][N];

float verify[N][N];

uint64_t nanos(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000 + (uint64_t)t.tv_nsec;
}

int main(){

    FILE *f = fopen("/tmp/gemm", "rb");
    if (f == NULL) {
        printf("Error opening file!\n");
        return -1;
    }
    fread(A, 1, sizeof(float)*N*N, f);
    fread(B, 1, sizeof(float)*N*N, f);
    fread(verify, 1, sizeof(float)*N*N, f);
    fclose(f);


    double flops = 2.0 * N * N * N * 1e-9;
    uint64_t start = nanos();

    // perform matrix multplication
    for(int x = 0; x < N; x++){
        for(int y = 0; y < N; y++){
            float acc = 0;
            for(int z = 0; z < N; z++){
                acc += A[x][z] * B[z][y];
            }
            C[x][y] = acc;
        }
    }
    
    uint64_t end = nanos();

    // printf("Time: %fs\n", (end - start)*1e-9);
    double gflops = (double)flops / (double)((end - start) * 1e-9);
    printf("%f GFLOP/S \n", gflops);

    for (int k = 0; k < N*N; k++) {
        if (fabsf(C[0][k]- verify[0][k]) > 1e-3) {
            printf("Verification failed\n");
            return 1;
        }
    }
    printf("Verfication successful\n");

    return 0;
}
