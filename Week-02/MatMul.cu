#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>


// MatMul native
__global__ void MatMul(float *a, float *b, float *c, int m, int n, int k) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n && row < m) { 
        float sum = 0;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
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

    // Launch kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (m + blockSize.y - 1) / blockSize.y);
    MatMul<<<gridSize, blockSize>>>(d_a, d_b, d_c, m, n, k);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("MatMul kernel failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(c, d_c, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // reference
    MatMul_ref(a, b, c_ref, m, n, k);

    // Verify results
    for (int i = 0; i < m * n; i++) {
        if (abs(c[i] - c_ref[i]) > 1e-5) {
            printf("Error: c[%d] = %f, c_ref[%d] = %f\n", i, c[i], i, c_ref[i]);
            return -1;
        }
    }

    printf("Success!\n");

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    free(a);
    free(b);
    free(c);
    free(c_ref);

    return 0;
}
