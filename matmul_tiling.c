// clang -O3 matmul_tiling.c && ./a.out
// ~5.8 GFLOP/s
#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

#define N 1024
#define BLOCK_SIZE 8

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

    for (int bx = 0; bx < N; bx += BLOCK_SIZE){
        for (int by = 0; by < N; by += BLOCK_SIZE){

                for (int x = bx; x < bx + BLOCK_SIZE; x++){
                    for (int y = by; y < by + BLOCK_SIZE; y++){
                        float acc = 0;
                        for (int k = 0; k < N; k++){
                            acc += A[x][k] * B[k][y];
                        }
                        C[x][y] = acc;
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
