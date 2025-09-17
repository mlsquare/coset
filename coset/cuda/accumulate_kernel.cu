#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <c10/cuda/CUDAGuard.h>

// CUDA kernel for carry-aware accumulation
__global__ void carry_aware_accumulate_kernel(
    const int8_t* encodings,       // [M, batch_size, d]
    int64_t* layer_sums,           // [M, batch_size, d]
    int batch_size,
    int M,
    int q,
    int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int layer = blockIdx.y;
    
    if (idx >= batch_size || layer >= M) return;
    
    // Add encoding to layer sum
    for (int k = 0; k < d; k++) {
        int8_t val = encodings[layer * batch_size * d + idx * d + k];
        layer_sums[layer * batch_size * d + idx * d + k] += val;
    }
}

// CUDA kernel for normalization with carry propagation
__global__ void normalize_with_carry_kernel(
    int64_t* layer_sums,           // [M, batch_size, d]
    int8_t* normalized_encodings,  // [M, batch_size, d]
    int batch_size,
    int M,
    int q,
    int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    
    if (idx >= batch_size || k >= d) return;
    
    // Normalize with carry propagation
    for (int i = 0; i < M - 1; i++) {
        int64_t sum = layer_sums[i * batch_size * d + idx * d + k];
        
        // Compute carry
        int64_t carry = sum / q;
        
        // Adjust current layer
        int64_t adjusted = sum % q;
        layer_sums[i * batch_size * d + idx * d + k] = adjusted;
        
        // Add carry to next layer
        layer_sums[(i + 1) * batch_size * d + idx * d + k] += carry;
    }
    
    // Convert to int8
    for (int i = 0; i < M; i++) {
        int64_t sum = layer_sums[i * batch_size * d + idx * d + k];
        normalized_encodings[i * batch_size * d + idx * d + k] = (int8_t)(sum % q);
    }
}

// CUDA kernel for fast mod-q accumulation
__global__ void fast_modq_accumulate_kernel(
    int64_t* acc,                  // [batch_size, d]
    const int8_t* x,               // [batch_size, d]
    int q,
    int batch_size,
    int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    
    if (idx >= batch_size || k >= d) return;
    
    // Fast mod-q accumulation
    int64_t val = acc[idx * d + k] + x[idx * d + k];
    acc[idx * d + k] = val % q;
}

// CUDA kernel for batch accumulation
__global__ void batch_accumulate_kernel(
    const int8_t* encodings_batch, // [batch_size, M, d]
    int64_t* layer_sums,           // [M, batch_size, d]
    int batch_size,
    int M,
    int q,
    int d
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int layer = blockIdx.y;
    int k = blockIdx.z;
    
    if (idx >= batch_size || layer >= M || k >= d) return;
    
    // Add encoding to layer sum
    int8_t val = encodings_batch[idx * M * d + layer * d + k];
    layer_sums[layer * batch_size * d + idx * d + k] += val;
}

// CUDA kernel for periodic normalization
__global__ void periodic_normalize_kernel(
    int64_t* layer_sums,           // [M, batch_size, d]
    int8_t* normalized_encodings,  // [M, batch_size, d]
    int batch_size,
    int M,
    int q,
    int d,
    int normalize_frequency
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y;
    
    if (idx >= batch_size || k >= d) return;
    
    // Check if it's time to normalize
    if ((idx % normalize_frequency) == 0) {
        // Normalize with carry propagation
        for (int i = 0; i < M - 1; i++) {
            int64_t sum = layer_sums[i * batch_size * d + idx * d + k];
            
            // Compute carry
            int64_t carry = sum / q;
            
            // Adjust current layer
            int64_t adjusted = sum % q;
            layer_sums[i * batch_size * d + idx * d + k] = adjusted;
            
            // Add carry to next layer
            layer_sums[(i + 1) * batch_size * d + idx * d + k] += carry;
        }
    }
    
    // Convert to int8
    for (int i = 0; i < M; i++) {
        int64_t sum = layer_sums[i * batch_size * d + idx * d + k];
        normalized_encodings[i * batch_size * d + idx * d + k] = (int8_t)(sum % q);
    }
}

// Host function to launch carry-aware accumulation kernel
void cuda_carry_aware_accumulate(
    const torch::Tensor& encodings,
    torch::Tensor& layer_sums,
    int q,
    int d
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(encodings.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Get dimensions
    int M = encodings.size(0);
    int batch_size = encodings.size(1);
    
    // Launch kernel
    dim3 threads_per_block(256, 1);
    dim3 num_blocks((batch_size + threads_per_block.x - 1) / threads_per_block.x, M);
    
    carry_aware_accumulate_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        encodings.data_ptr<int8_t>(),
        layer_sums.data_ptr<int64_t>(),
        batch_size,
        M,
        q,
        d
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// Host function to launch normalization with carry kernel
torch::Tensor cuda_normalize_with_carry(
    torch::Tensor& layer_sums,
    int q,
    int d
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(layer_sums.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Get dimensions
    int M = layer_sums.size(0);
    int batch_size = layer_sums.size(1);
    
    // Create output tensor
    auto normalized_encodings = torch::zeros({M, batch_size, d}, 
        torch::TensorOptions().dtype(torch::kInt8).device(layer_sums.device()));
    
    // Launch kernel
    dim3 threads_per_block(256, 1);
    dim3 num_blocks((batch_size + threads_per_block.x - 1) / threads_per_block.x, d);
    
    normalize_with_carry_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        layer_sums.data_ptr<int64_t>(),
        normalized_encodings.data_ptr<int8_t>(),
        batch_size,
        M,
        q,
        d
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return normalized_encodings;
}

// Host function to launch fast mod-q accumulation kernel
void cuda_fast_modq_accumulate(
    torch::Tensor& acc,
    const torch::Tensor& x,
    int q
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(acc.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Get dimensions
    int batch_size = acc.size(0);
    int d = acc.size(1);
    
    // Launch kernel
    dim3 threads_per_block(256, 1);
    dim3 num_blocks((batch_size + threads_per_block.x - 1) / threads_per_block.x, d);
    
    fast_modq_accumulate_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        acc.data_ptr<int64_t>(),
        x.data_ptr<int8_t>(),
        q,
        batch_size,
        d
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// Host function to launch batch accumulation kernel
void cuda_batch_accumulate(
    const torch::Tensor& encodings_batch,
    torch::Tensor& layer_sums,
    int q,
    int d
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(encodings_batch.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Get dimensions
    int batch_size = encodings_batch.size(0);
    int M = encodings_batch.size(1);
    
    // Launch kernel
    dim3 threads_per_block(256, 1, 1);
    dim3 num_blocks(
        (batch_size + threads_per_block.x - 1) / threads_per_block.x,
        M,
        d
    );
    
    batch_accumulate_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        encodings_batch.data_ptr<int8_t>(),
        layer_sums.data_ptr<int64_t>(),
        batch_size,
        M,
        q,
        d
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}

// Host function to launch periodic normalization kernel
torch::Tensor cuda_periodic_normalize(
    torch::Tensor& layer_sums,
    int q,
    int d,
    int normalize_frequency
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(layer_sums.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Get dimensions
    int M = layer_sums.size(0);
    int batch_size = layer_sums.size(1);
    
    // Create output tensor
    auto normalized_encodings = torch::zeros({M, batch_size, d}, 
        torch::TensorOptions().dtype(torch::kInt8).device(layer_sums.device()));
    
    // Launch kernel
    dim3 threads_per_block(256, 1);
    dim3 num_blocks((batch_size + threads_per_block.x - 1) / threads_per_block.x, d);
    
    periodic_normalize_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        layer_sums.data_ptr<int64_t>(),
        normalized_encodings.data_ptr<int8_t>(),
        batch_size,
        M,
        q,
        d,
        normalize_frequency
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return normalized_encodings;
}
