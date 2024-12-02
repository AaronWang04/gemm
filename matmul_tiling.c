// clang -O3 matmul_tiling.c && ./a.out
// ~6 GFLOP/s with transpose, ~1 GFLOP/s without transpose
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

#define N 1024
#define BLOCK_SIZE 8

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
    assert (N % BLOCK_SIZE == 0);

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
                AT[i + l][j] = A[j][i + l];
            }
        }
    }

    // ~36 GFLOP/s

    // around 36 gflops on block size 8, slower on any other block size
    for (int bx = 0; bx < N; bx += BLOCK_SIZE){
        for (int by = 0; by < N; by += BLOCK_SIZE){

            for (int k = 0; k < N; k++){
                for (int x = bx; x < bx + BLOCK_SIZE; x++){
                    for (int y = by; y < by + BLOCK_SIZE; y++){
                        C[x][y] += A[x][k] * BT[y][k];
                    }
                }
            }

        }
    }

    // around 30 gflops on block size 128, slower on any other block size
    // for (int bx = 0; bx < N; bx += BLOCK_SIZE){
    //     for (int by = 0; by < N; by += BLOCK_SIZE){

    //         for (int k = 0; k < N; k++){
    //             for (int x = bx; x < bx + BLOCK_SIZE; x++){
    //                 for (int y = by; y < by + BLOCK_SIZE; y++){
    //                     C[x][y] += AT[k][x] * B[k][y];
    //                 }
    //             }
    //         }

    //     }
    // }

    // ~6 GFLOP/s im not sure why this way doesnt work at all
    // for (int bx = 0; bx < N; bx += BLOCK_SIZE){
    //     for (int by = 0; by < N; by += BLOCK_SIZE){

    //         for (int x = bx; x < bx + BLOCK_SIZE; x++){
    //             for (int y = by; y < by + BLOCK_SIZE; y++){
    //                 for (int k = 0; k < N; k++){
    //                     C[x][y] += A[x][k] * BT[y][k];
    //                 }
    //             }
    //         }

    //     }
    // }

    uint64_t end = nanos();

    printf("Time: %fs\n", (end - start)*1e-9);
    // calculate flops
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
