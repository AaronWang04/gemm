// nvcc -std=c++17 -o matrix_transpose matrix_transpose.cu && ./matrix_transpose

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define BLOCK_ROWS 8

/*
matrix = [
             1,  2,  3,  4,
             5,  6,  7,  8,
             9, 10, 11, 12,
            13, 14, 15, 16
        ]

output_matrix = [
             1,  5,   9, 13,
             2,  6,  10, 14,
             3,  7,  11, 15,
             4,  8,  12, 16
        ]

1. host sends data from vector to HBM
2. data is in HBM and being fetched by the SM that dispatched the threadblock

H100 -> some number of SMs
the SM dispatches a threadblock to work on data

total lds: 9 -> each are 100 cycles
total strs: 9 -> each are 100 cycles
*/

__global__ void transpose_kernel(const float* input, float* output, int width, int height) {
    // int index = blockIdx.x * blockDim.x + threadIdx.x; 
    // int total_elements = width * height;

    __shared__ float tile[TILE_SIZE][TILE_SIZE+1];
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; 
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;

    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS) {
        if (col < width && (row + i) < height){
            tile[threadIdx.y + i][threadIdx.x] = input[(row + i) * width + col];
        }
    }
    __syncthreads();

    col = blockIdx.y * TILE_SIZE + threadIdx.x;
    row = blockIdx.x * TILE_SIZE + threadIdx.y;
    for (int i = 0; i < TILE_SIZE; i += BLOCK_ROWS) {
        if (col < height && (row + i) < width){
            output[(row + i) * height + col] = tile[threadIdx.x][threadIdx.y + i];
        }
    }

    // if (index < total_elements) {
    //     int row = index / width;
    //     int col = index % width;
    //     output[col * height + row] = input[row * width + col];
    // }
}


void transpose_cpu(const thrust::host_vector<float>& input, thrust::host_vector<float>& output, int width, int height) {
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            output[col * height + row] = input[row * width + col];
        }
    }
}

int main() {
    const int width = 1024;
    const int height = 1024;
    const int num_iterations = 100;

    thrust::host_vector<float> h_input(width * height);
    thrust::host_vector<float> h_output_cpu(width * height);
    thrust::host_vector<float> h_output_gpu(width * height);

    std::srand(static_cast<unsigned int>(std::time(0)));
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = static_cast<float>(std::rand()) / RAND_MAX;
    }

    transpose_cpu(h_input, h_output_cpu, width, height);

    thrust::device_vector<float> d_input = h_input;
    thrust::device_vector<float> d_output(width * height);

    dim3 blockSize(TILE_SIZE, BLOCK_ROWS);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE, (height + TILE_SIZE - 1) / TILE_SIZE);

    // Warm-up kernel launch to initialize the CUDA context
    transpose_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()), width, height);
    cudaDeviceSynchronize();

    // Timing the kernel over multiple iterations
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    for (int i = 0; i < num_iterations; ++i) {
        transpose_kernel<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()), width, height);
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    float average_time = milliseconds / num_iterations;

    h_output_gpu = d_output;

    bool match = true;
    for (int i = 0; i < width * height; ++i) {
        if (h_output_cpu[i] != h_output_gpu[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "Transpose successful." << std::endl;
        std::cout << "Average kernel execution time over " << num_iterations << " iterations: " << average_time << " ms" << std::endl;
    } else {
        std::cout << "Transpose failed." << std::endl;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
