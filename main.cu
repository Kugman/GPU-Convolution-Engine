
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <vector>
#include <stdio.h>

#define N 5
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

__global__ void conv2D_GPU(float *input, float *kernel, float* output, int n, int k) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int pad = k / 2;

    if (row < pad || row >= n - pad || col < pad || col >= n - pad) return;

    float sum = 0.0f;
    
    for (int i = -pad; i <= pad; i++) 
        for (int j = -pad; j <= pad; j++)
            sum += input[(row + i) * n + (col + j)] * kernel[(i + pad) * k + (j + pad)];

    output[row * n + col] = sum;
    
}

void cpyToHost(const Matrix &m, float* h, const int size) {
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            h[i * size + j] = m[i][j];    
}

int main()
{
    Matrix input = {
        {1, 2, 3, 4, 5},
        {5, 6, 7, 8, 9},
        {9, 10, 11, 12, 13},
        {13, 14, 15, 16, 17},
        {17, 18, 19, 20, 21}
    };

    Matrix kernel = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };

    Matrix cpu_output = conv2D_CPU(input, kernel);

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
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y) / threadsPerBlock.y);
    
    conv2D_GPU <<<numBlocks, threadsPerBlock>>> (d_input, d_kernel, d_output, N, K);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);

    cout << "GPU Convolution output : " << endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            cout << h_output[i * N + j] << "\t";
        cout << endl;
    }


    
    cout << "CPU Convolution output: " << endl;
    for (auto& row : cpu_output) {
        for (auto& val : row)
            cout << val << "\t";
        cout << endl;
    }


    return 0;
}