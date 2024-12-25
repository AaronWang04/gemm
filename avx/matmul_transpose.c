// clang -O3 avx/matmul_transpose.c && ./a.out
// ~6 GFLOP/s
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

#define N 1024

float A[N][N];
float AT[N][N];
float B[N][N];
float BT[N][N];
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
    for (int i = 0; i < N; i+=8) {
        for (int j = 0; j < N; j++) {
            for (int l = 0; l < 8; l++) {
                BT[i + l][j] = B[j][i + l];
                // AT[i + l][j] = A[j][i + l];
            }
        }
    }

    for(int x = 0; x < N; x++){
        for(int y = 0; y < N; y++){
            float acc = 0;
            for(int z = 0; z < N; z++){
                acc += A[x][z] * BT[y][z];
            }
            C[x][y] = acc;
        }
    }

    // around 0.7 flops
    // for(int x = 0; x < N; x++){
    //     for(int y = 0; y < N; y++){
    //         float acc = 0;
    //         for(int z = 0; z < N; z++){
    //             acc += AT[z][x] * B[z][y];
    //         }
    //         C[x][y] = acc;
    //     }
    // }
    
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
