#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <c10/cuda/CUDAGuard.h>

// CUDA kernel for LUT-based inner product
__global__ void lut_inner_product_kernel(
    const int8_t* encodings_x,     // [M, batch_size, d]
    const int8_t* encodings_y,     // [M, batch_size, d]
    const int32_t* lut,            // [lut_size, lut_size]
    int64_t* results,              // [batch_size]
    int batch_size,
    int M,
    int q,
    int d,
    int lut_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    int64_t result = 0;
    
    // Compute inner product using LUT
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            // Convert encoding to LUT index
            int idx_x = 0;
            int idx_y = 0;
            
            for (int k = 0; k < d; k++) {
                int8_t val_x = encodings_x[i * batch_size * d + idx * d + k];
                int8_t val_y = encodings_y[j * batch_size * d + idx * d + k];
                
                idx_x = idx_x * q + val_x;
                idx_y = idx_y * q + val_y;
            }
            
            // Look up in LUT
            int32_t lut_value = lut[idx_x * lut_size + idx_y];
            
            // Scale by q^(i+j)
            int64_t scale_factor = 1;
            for (int s = 0; s < i + j; s++) {
                scale_factor *= q;
            }
            
            result += lut_value * scale_factor;
        }
    }
    
    results[idx] = result;
}

// CUDA kernel for MAC operations in encoding space
__global__ void mac_encoding_space_kernel(
    const int8_t* encodings_x,     // [M, batch_size, d]
    const int8_t* encodings_y,     // [M, batch_size, d]
    const int32_t* lut,            // [lut_size, lut_size]
    int64_t* results,              // [batch_size]
    int batch_size,
    int M,
    int q,
    int d,
    int lut_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    int64_t result = 0;
    
    // Compute MAC using LUT
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            // Convert encoding to LUT index
            int idx_x = 0;
            int idx_y = 0;
            
            for (int k = 0; k < d; k++) {
                int8_t val_x = encodings_x[i * batch_size * d + idx * d + k];
                int8_t val_y = encodings_y[j * batch_size * d + idx * d + k];
                
                idx_x = idx_x * q + val_x;
                idx_y = idx_y * q + val_y;
            }
            
            // Look up in LUT
            int32_t lut_value = lut[idx_x * lut_size + idx_y];
            
            // Scale by q^(i+j)
            int64_t scale_factor = 1;
            for (int s = 0; s < i + j; s++) {
                scale_factor *= q;
            }
            
            result += lut_value * scale_factor;
        }
    }
    
    results[idx] = result;
}

// CUDA kernel for batch MAC operations
__global__ void batch_mac_kernel(
    const int8_t* encodings_batch_x,  // [batch_size, M, d]
    const int8_t* encodings_batch_y,  // [batch_size, M, d]
    const int32_t* lut,               // [lut_size, lut_size]
    int64_t* results,                 // [batch_size]
    int batch_size,
    int M,
    int q,
    int d,
    int lut_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    int64_t result = 0;
    
    // Compute MAC for this batch element
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            // Convert encoding to LUT index
            int idx_x = 0;
            int idx_y = 0;
            
            for (int k = 0; k < d; k++) {
                int8_t val_x = encodings_batch_x[idx * M * d + i * d + k];
                int8_t val_y = encodings_batch_y[idx * M * d + j * d + k];
                
                idx_x = idx_x * q + val_x;
                idx_y = idx_y * q + val_y;
            }
            
            // Look up in LUT
            int32_t lut_value = lut[idx_x * lut_size + idx_y];
            
            // Scale by q^(i+j)
            int64_t scale_factor = 1;
            for (int s = 0; s < i + j; s++) {
                scale_factor *= q;
            }
            
            result += lut_value * scale_factor;
        }
    }
    
    results[idx] = result;
}

// CUDA kernel for adaptive MAC with early exit
__global__ void adaptive_mac_kernel(
    const int8_t* encodings_x,     // [M, batch_size, d]
    const int8_t* encodings_y,     // [M, batch_size, d]
    const int32_t* lut,            // [lut_size, lut_size]
    int64_t* results,              // [batch_size]
    int batch_size,
    int M,
    int max_layers,
    int q,
    int d,
    int lut_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    int64_t result = 0;
    
    // Compute MAC with early exit
    int actual_layers = min(M, max_layers);
    
    for (int i = 0; i < actual_layers; i++) {
        for (int j = 0; j < actual_layers; j++) {
            // Convert encoding to LUT index
            int idx_x = 0;
            int idx_y = 0;
            
            for (int k = 0; k < d; k++) {
                int8_t val_x = encodings_x[i * batch_size * d + idx * d + k];
                int8_t val_y = encodings_y[j * batch_size * d + idx * d + k];
                
                idx_x = idx_x * q + val_x;
                idx_y = idx_y * q + val_y;
            }
            
            // Look up in LUT
            int32_t lut_value = lut[idx_x * lut_size + idx_y];
            
            // Scale by q^(i+j)
            int64_t scale_factor = 1;
            for (int s = 0; s < i + j; s++) {
                scale_factor *= q;
            }
            
            result += lut_value * scale_factor;
        }
    }
    
    results[idx] = result;
}

// Host function to launch LUT inner product kernel
torch::Tensor cuda_lut_inner_product(
    const torch::Tensor& encodings_x,
    const torch::Tensor& encodings_y,
    const torch::Tensor& lut,
    int q,
    int d
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(encodings_x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Get dimensions
    int M = encodings_x.size(0);
    int batch_size = encodings_x.size(1);
    int lut_size = lut.size(0);
    
    // Create output tensor
    auto results = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt64).device(encodings_x.device()));
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    lut_inner_product_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        encodings_x.data_ptr<int8_t>(),
        encodings_y.data_ptr<int8_t>(),
        lut.data_ptr<int32_t>(),
        results.data_ptr<int64_t>(),
        batch_size,
        M,
        q,
        d,
        lut_size
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return results;
}

// Host function to launch MAC encoding space kernel
torch::Tensor cuda_mac_encoding_space(
    const torch::Tensor& encodings_x,
    const torch::Tensor& encodings_y,
    const torch::Tensor& lut,
    int q,
    int d
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(encodings_x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Get dimensions
    int M = encodings_x.size(0);
    int batch_size = encodings_x.size(1);
    int lut_size = lut.size(0);
    
    // Create output tensor
    auto results = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt64).device(encodings_x.device()));
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    mac_encoding_space_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        encodings_x.data_ptr<int8_t>(),
        encodings_y.data_ptr<int8_t>(),
        lut.data_ptr<int32_t>(),
        results.data_ptr<int64_t>(),
        batch_size,
        M,
        q,
        d,
        lut_size
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return results;
}

// Host function to launch batch MAC kernel
torch::Tensor cuda_batch_mac(
    const torch::Tensor& encodings_batch_x,
    const torch::Tensor& encodings_batch_y,
    const torch::Tensor& lut,
    int q,
    int d
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(encodings_batch_x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Get dimensions
    int batch_size = encodings_batch_x.size(0);
    int M = encodings_batch_x.size(1);
    int lut_size = lut.size(0);
    
    // Create output tensor
    auto results = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt64).device(encodings_batch_x.device()));
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    batch_mac_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        encodings_batch_x.data_ptr<int8_t>(),
        encodings_batch_y.data_ptr<int8_t>(),
        lut.data_ptr<int32_t>(),
        results.data_ptr<int64_t>(),
        batch_size,
        M,
        q,
        d,
        lut_size
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return results;
}

// Host function to launch adaptive MAC kernel
torch::Tensor cuda_adaptive_mac(
    const torch::Tensor& encodings_x,
    const torch::Tensor& encodings_y,
    const torch::Tensor& lut,
    int max_layers,
    int q,
    int d
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(encodings_x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Get dimensions
    int M = encodings_x.size(0);
    int batch_size = encodings_x.size(1);
    int lut_size = lut.size(0);
    
    // Create output tensor
    auto results = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt64).device(encodings_x.device()));
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    adaptive_mac_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        encodings_x.data_ptr<int8_t>(),
        encodings_y.data_ptr<int8_t>(),
        lut.data_ptr<int32_t>(),
        results.data_ptr<int64_t>(),
        batch_size,
        M,
        max_layers,
        q,
        d,
        lut_size
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return results;
}
