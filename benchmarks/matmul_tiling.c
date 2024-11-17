#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

#define N 1024
#define BLOCK_SIZE 32

float A[N][N];
float B[N][N];
float C[N][N];

uint64_t nanos(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000 + (uint64_t)t.tv_nsec;
}

int main(){
    assert (N % BLOCK_SIZE == 0);
    uint64_t start = nanos();

    double flops = 2.0 * N * N * N * 1e-9;

    for (int bi = 0; bi < N; bi += BLOCK_SIZE){
        for (int bj = 0; bj < N; bj += BLOCK_SIZE){
            for (int bk = 0; bk < N; bk += BLOCK_SIZE){
                for (int i = bi; i < bi + BLOCK_SIZE; i++){
                    for (int j = bj; j < bj + BLOCK_SIZE; j++){
                        for (int k = bk; k < bk + BLOCK_SIZE; k++){
                            C[i][j] += A[i][k] * B[k][j];
                        }
                    }
                }
            }
        }
    }
    uint64_t end = nanos();

    printf("Time: %fs\n", (end - start)*1e-9);
    // calculate flops
    double gflops = (double)flops / (double)((end - start) * 1e-9);
    printf("GFLOPS: %f\n", gflops);
    return 0;
}
