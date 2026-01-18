#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

template<typename T>
__global__ void matmul(const T* A, const T* B, T* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        T sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

template<typename T>
void run_benchmark(int N, const char* type_name) {
    size_t bytes = N * N * sizeof(T);
    
    T *h_A = (T*)malloc(bytes);
    T *h_B = (T*)malloc(bytes);
    T *h_C = (T*)malloc(bytes);
    
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = 1;
        h_B[i] = 1;
    }
    
    T *d_A, *d_B, *d_C;
    checkCuda(cudaMalloc(&d_A, bytes));
    checkCuda(cudaMalloc(&d_B, bytes));
    checkCuda(cudaMalloc(&d_C, bytes));
    
    checkCuda(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    
    /*matmul<<<grid, block>>>(d_A, d_B, d_C, N);
    checkCuda(cudaDeviceSynchronize());*/
    
    auto start = std::chrono::high_resolution_clock::now();
    matmul<<<grid, block>>>(d_A, d_B, d_C, N);
    checkCuda(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    printf("%-6s N=%4d  Zeit=%.3f ms\n", type_name, N, ms);
    
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
}

int main() {
    printf("CUDA Matrixmultiplikation\n");
    printf("=========================\n");
    
    run_benchmark<float>(2048, "float");
    run_benchmark<double>(2048, "double");
    run_benchmark<float>(4096, "float");
    run_benchmark<double>(4096, "double");
    run_benchmark<float>(8192, "float");
    run_benchmark<double>(8192, "double");
    run_benchmark<float>(16384, "float");
    run_benchmark<double>(16384, "double");
    
    checkCuda(cudaDeviceReset());
    return 0;
}
