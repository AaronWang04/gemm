#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include <assert.h>

#define N 1024

float A[N][N];
float B[N][N];
float C[N][N];

uint64_t nanos(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return (uint64_t)t.tv_sec * 1000000000 + (uint64_t)t.tv_nsec;
}

int main(){
    uint64_t start = nanos();

    double flops = 2.0 * N * N * N * 1e-9;
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
    return 0;
}
