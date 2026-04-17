#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// vecAdd func
__global__ void vecAdd(float *a, float *b, float *c, int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ref program
void vecAdd_ref(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <vector_size>\n", argv[0]);
        return -1;
    }

    // vec size
    int n = atoi(argv[1]);

    if (n <= 0) {
        printf("Error: Vector size is smaller than 0\n");
        return -1;
    }
    printf("Vector size: %d\n", n);

    float *a_h = (float*)malloc(n * sizeof(float));
    float *b_h = (float*)malloc(n * sizeof(float));
    float *c_h = (float*)malloc(n * sizeof(float));
    float *c_ref = (float*)malloc(n * sizeof(float));

    // init vec with random values
    for (int i = 0; i < n; i++) {
        a_h[i] = rand();
        b_h[i] = rand();
    }

    // alloc device memory
    float *a_d, *b_d, *c_d;
    cudaMalloc((void**)&a_d, n * sizeof(float));
    cudaMalloc((void**)&b_d, n * sizeof(float));
    cudaMalloc((void**)&c_d, n * sizeof(float));

    // copy data from host to device
    cudaMemcpy(a_d, a_h, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, n * sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // launch kernel
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(a_d, b_d, c_d, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("VecAdd kernel failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    cudaDeviceSynchronize();

    // copy res to host
    cudaMemcpy(c_h, c_d, n * sizeof(float), cudaMemcpyDeviceToHost);

    // ref program
    vecAdd_ref(a_h, b_h, c_ref, n);

    bool res_correct = true;
    for (int i = 0; i < n; i++) {
        if (abs(c_h[i] - c_ref[i]) > 1e-5) {
            res_correct = false;
            printf("Result verification failed at %d: CPU %.6f, GPU %.6f\n", i, c_ref[i], c_h[i]);
        }
    }

    if (res_correct) {
        printf("Vector addition successfully.\n");
    } else {
        printf("Vector addition failed.\n");
    }

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    free(a_h);
    free(b_h);
    free(c_h);

    return 0;
}