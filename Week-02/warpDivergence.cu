#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// kernel with warp divergence
__global__ void warpDivergence(float *a, float *b, float *c, int n, int array_size) { 
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < array_size){
        if (i < n) {
            c[i] = a[i] + b[i];
        }
        else {
            c[i] = 0; // This will cause warp divergence for threads with i >= n
        }
    }        
}



int main(int argc, char **argv) { 

    if (argc != 2) {
        printf("Usage: %s [array size]\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int array_size = 1024;
    if (n > array_size) {
        printf("Array size must be less than %d\n", array_size);
        return 1;
    }
    
    // Allocate host memory
    float *a = (float *)malloc(array_size * sizeof(float));
    float *b = (float *)malloc(array_size * sizeof(float));
    float *c = (float *)malloc(array_size * sizeof(float));

    // Initialize arrays
    for (int i = 0; i < array_size; i++) {
        a[i] = i;
        b[i] = array_size - i;
    }

    // Allocate device memory
    float *a_d, *b_d, *c_d;
    cudaMalloc((void **)&a_d, array_size * sizeof(float));
    cudaMalloc((void **)&b_d, array_size * sizeof(float));
    cudaMalloc((void **)&c_d, array_size * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(a_d, a, array_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b, array_size * sizeof(float), cudaMemcpyHostToDevice);

    // create cuda event
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warp up
    for (int i = 0; i < 5; i++){
        warpDivergence<<<1, 1024>>>(a_d, b_d, c_d, n, array_size);
    }

    cudaEventRecord(start);

    for (int i = 0; i < 100; i++){
        warpDivergence<<<1, 1024>>>(a_d, b_d, c_d, n, array_size);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {   
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // copy result back to host
    cudaMemcpy(c, c_d, array_size * sizeof(float), cudaMemcpyDeviceToHost);

    float gpu_time_ms = 0.0f;
    cudaEventElapsedTime(&gpu_time_ms, start, stop);

    printf("GPU time: %.2f ms\n", gpu_time_ms);

    // release device memory
    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    free(a);
    free(b);
    free(c);

    return 0;

}
