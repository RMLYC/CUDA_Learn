#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


// MatMul tile
#define TILE_SIZE 16

__global__ void MatMul(float *a, float *b, float *c, int m, int n, int k) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // allocate shared memory
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {

        // load tile to shared memory
        int a_offset = row * k + t * TILE_SIZE + threadIdx.x;
        int b_offset = (t * TILE_SIZE + threadIdx.y) * n + col;

        // load data a
        if (row < m && t * TILE_SIZE + threadIdx.x < k){
            As[threadIdx.y][threadIdx.x] = a[a_offset];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // load data b
        if (col < n && t * TILE_SIZE + threadIdx.y < k){
            Bs[threadIdx.y][threadIdx.x] = b[b_offset];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // compute
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }
    
    // 添加边界检查以防止写入超出矩阵范围
    if(row < m && col < n) {
        c[row * n + col] = sum;
    }
}

// MatMul reference
void MatMul_ref(float *a, float *b, float *c, int m, int n, int k) { 
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0;
            for (int l = 0; l < k; l++) {
                sum += a[i * k + l] * b[l * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

int main(int argc, char **argv) { 
    if (argc != 4) {
        printf("Usage: %s <m> <n> <k>\n", argv[0]);
        return 1;
    }

    int m = atoi(argv[1]);
    int n = atoi(argv[2]);
    int k = atoi(argv[3]);

    // Allocate host memory
    float *a = (float *)malloc(m * k * sizeof(float));
    float *b = (float *)malloc(k * n * sizeof(float));
    float *c = (float *)malloc(m * n * sizeof(float));
    float *c_ref = (float *)malloc(m * n * sizeof(float));

    // Initialize matrices
    for (int i = 0; i < m * k; i++) {
        a[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < k * n; i++) {
        b[i] = rand() / (float)RAND_MAX;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, m * k * sizeof(float));
    cudaMalloc(&d_b, k * n * sizeof(float));
    cudaMalloc(&d_c, m * n * sizeof(float));

    // Copy matrices to device
    cudaMemcpy(d_a, a, m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, k * n * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for kernel timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    cudaEventRecord(start);
    MatMul<<<gridSize, blockSize>>>(d_a, d_b, d_c, m, n, k);
    cudaEventRecord(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("MatMul kernel failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    // Copy result back to host
    cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // reference
    clock_t cpu_start = clock();
    MatMul_ref(a, b, c_ref, m, n, k);
    clock_t cpu_end = clock();
    double cpu_time_ms = (double)(cpu_end - cpu_start) * 1000.0 / CLOCKS_PER_SEC;

    // Verify results
    for (int i = 0; i < m * n; i++) {
        if (fabsf(c[i] - c_ref[i]) > 1e-4f) {
            printf("Error: c[%d] = %f, c_ref[%d] = %f\n", i, c[i], i, c_ref[i]);
            return -1;
        }
    }

    printf("Success!\n");
    printf("GPU kernel time: %.3f ms\n", gpu_time_ms);
    printf("CPU reference time: %.3f ms\n", cpu_time_ms);

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Free host memory
    free(a);
    free(b);
    free(c);
    free(c_ref);

    return 0;
}
