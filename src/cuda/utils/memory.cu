#include <torch/extension.h>
#include <cuda_runtime.h>

// Memory management utilities for CUDA kernels

__global__ void fill_zeros_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 0.0f;
    }
}

__global__ void copy_tensor_kernel(const float* src, float* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = src[idx];
    }
}

torch::Tensor fill_zeros_cuda(torch::Tensor tensor) {
    int size = tensor.numel();
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    fill_zeros_kernel<<<blocks, threads_per_block>>>(
        tensor.data_ptr<float>(),
        size
    );
    
    cudaDeviceSynchronize();
    return tensor;
}

torch::Tensor copy_tensor_cuda(torch::Tensor src, torch::Tensor dst) {
    int size = src.numel();
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    copy_tensor_kernel<<<blocks, threads_per_block>>>(
        src.data_ptr<float>(),
        dst.data_ptr<float>(),
        size
    );
    
    cudaDeviceSynchronize();
    return dst;
}
