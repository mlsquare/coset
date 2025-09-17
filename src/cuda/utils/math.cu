#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Mathematical utility functions for CUDA kernels

__device__ float fast_exp_cuda(float x) {
    // Fast exponential approximation
    const float c1 = 0.6931471805599453f;
    const float c2 = 1.4426950408889634f;
    const float c3 = 1.0f / 120.0f;
    const float c4 = 1.0f / 24.0f;
    const float c5 = 1.0f / 6.0f;
    
    float y = x * c2;
    float n = floorf(y);
    float f = y - n;
    
    float r = 1.0f + f + f * f * c5 + f * f * f * c4 + f * f * f * f * c3;
    
    // Scale by 2^n
    int exp = (int)n;
    if (exp > 127) return 3.4028235e38f;
    if (exp < -127) return 0.0f;
    
    return r * __expf(c1 * n);
}

__device__ float fast_log_cuda(float x) {
    // Fast logarithm approximation
    if (x <= 0.0f) return -3.4028235e38f;
    
    const float c1 = 0.6931471805599453f;
    const float c2 = 1.4426950408889634f;
    
    int exp;
    float mantissa = frexpf(x, &exp);
    
    return c1 * (float)exp + (mantissa - 1.0f) * c2;
}

__global__ void elementwise_add_kernel(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] + b[idx];
    }
}

__global__ void elementwise_multiply_kernel(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] * b[idx];
    }
}

__global__ void elementwise_divide_kernel(
    const float* a,
    const float* b,
    float* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        result[idx] = a[idx] / b[idx];
    }
}

torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto result = torch::zeros_like(a);
    int size = a.numel();
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    elementwise_add_kernel<<<blocks, threads_per_block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        size
    );
    
    cudaDeviceSynchronize();
    return result;
}

torch::Tensor elementwise_multiply_cuda(torch::Tensor a, torch::Tensor b) {
    auto result = torch::zeros_like(a);
    int size = a.numel();
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    elementwise_multiply_kernel<<<blocks, threads_per_block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        size
    );
    
    cudaDeviceSynchronize();
    return result;
}

torch::Tensor elementwise_divide_cuda(torch::Tensor a, torch::Tensor b) {
    auto result = torch::zeros_like(a);
    int size = a.numel();
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    elementwise_divide_kernel<<<blocks, threads_per_block>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        size
    );
    
    cudaDeviceSynchronize();
    return result;
}
