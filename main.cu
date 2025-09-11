
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <vector>
#include <chrono>

#define N 200
#define K 3

using namespace std;
using Matrix = vector<vector<float>>;

Matrix conv2D_CPU(const Matrix& input, const Matrix& kernel) {
   
    int n = input.size(),
        k = kernel.size(),
        pad = k / 2;

    Matrix output(n, vector<float>(n, 0));

    for (int i = pad; i < n - pad; i++) {
        for (int j = pad; j < n - pad; j++) {
            float sum = 0.0f;
            for (int ki = -pad; ki <= pad; ki++) {
                for (int kj = -pad; kj <= pad; kj++) {
                    sum += input[i + ki][j + kj] * kernel[ki + pad][kj + pad];
                }
            }
            output[i][j] = sum;
        }
    }
    return output;
}

__global__ void conv2D_GPU(const float *__restrict__ input, const float * __restrict__ kernel, float *__restrict__ output, const int n, const int k) {
    
    extern __shared__ float shmem[];
    float* tile = shmem;
    float* shKernel = (float*)&tile[(blockDim.y + k - 1) * (blockDim.x + k - 1)];
    
    int pad = k / 2;
    int tile_size_x = blockDim.x;
    int tile_size_y = blockDim.y;

    int row = blockIdx.y * tile_size_y + threadIdx.y;
    int col = blockIdx.x * tile_size_x + threadIdx.x;

    int shared_row = threadIdx.y + pad;
    int shared_col = threadIdx.x + pad;  
    
    if (threadIdx.y < k && threadIdx.x < k)
        shKernel[threadIdx.y * k + threadIdx.x] = kernel[threadIdx.y * k + threadIdx.x];
    
    int global_row = row - pad;
    int global_col = col - pad;
    if (shared_row < tile_size_y + 2 * pad && shared_col < tile_size_x + 2 * pad) {
        if (global_row >= 0 && global_row < n && global_col >= 0 && global_col < n)
            tile[shared_row * (tile_size_x + 2 * pad) + shared_col] = input[global_row * n + global_col];
        else
            tile[shared_row * (tile_size_x + 2 * pad) + shared_col] = 0.0f;
    }

    __syncthreads();

    if (row < pad || row >= n - pad || col < pad || col >= n - pad) return;

    float sum = 0.0f;
    
    for (int i = -pad; i <= pad; i++) 
        for (int j = -pad; j <= pad; j++)
            sum += tile[(shared_row + i) * (tile_size_x + 2 * pad) + (shared_col + j)] * shKernel[(i + pad) * k + (j + pad)];

    output[row * n + col] = sum;
    
}

void cpyToHost(const Matrix &m, float* h, const int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            h[i * size + j] = m[i][j];    
}

int main()
{
    Matrix input(N, vector<float>(N));
    Matrix kernel = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            input[i][j] = i % 255;
    
    auto start_cpu = chrono::high_resolution_clock::now();
    Matrix cpu_output = conv2D_CPU(input, kernel);
    auto end_cpu = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> cpu_time = end_cpu - start_cpu;

    float h_input[N * N], h_kernel[K*K], h_output[N * N] = {0};
    cpyToHost(input, h_input, N);
    cpyToHost(kernel, h_kernel, K);


    float* d_input, * d_kernel, * d_output;

    cudaMalloc((void**)&d_input, N * N * sizeof(float));
    cudaMalloc((void**)&d_kernel, K * K * sizeof(float));
    cudaMalloc((void**)&d_output, N * N * sizeof(float));

    cudaMemcpy(d_input, h_input, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, K * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    int sharedMemSize = (threadsPerBlock.x + K - 1) * (threadsPerBlock.y + K - 1) * sizeof(float) + K * K * sizeof(float);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    conv2D_GPU <<<numBlocks, threadsPerBlock, sharedMemSize>>> (d_input, d_kernel, d_output, N, K);
    cudaEventRecord(stop);

    cudaMemcpy(h_output, d_output, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    float gpu_time = 0;
    cudaEventElapsedTime(&gpu_time, start, stop);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    cout << "GPU Time: " << gpu_time << " ms" << endl;
    //cout << "Convolution output : " << endl;
    //for (int i = 0; i < N; i++) {
    //    for (int j = 0; j < N; j++)
    //        cout << h_output[i * N + j] << "\t";
    //    cout << endl;
    //}
    //cout << endl;

    
    cout << "CPU Time: " << cpu_time.count() << " ms" << endl;
    //cout << "Convolution output : " << endl;
    //for (auto& row : cpu_output) {
    //    for (auto& val : row)
    //        cout << val << "\t";
    //    cout << endl;
    //}


    return 0;
}
