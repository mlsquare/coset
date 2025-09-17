#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <c10/cuda/CUDAGuard.h>

// CUDA kernel for building two-sided LUT
__global__ void build_lut_kernel(
    const float* lattice_points,   // [lut_size, d]
    int32_t* lut,                  // [lut_size, lut_size]
    int lut_size,
    int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= lut_size || j >= lut_size) return;
    
    // Compute inner product between lattice points
    float inner_product = 0.0f;
    for (int k = 0; k < d; k++) {
        inner_product += lattice_points[i * d + k] * lattice_points[j * d + k];
    }
    
    lut[i * lut_size + j] = (int32_t)inner_product;
}

// CUDA kernel for building one-sided LUT
__global__ void build_one_sided_lut_kernel(
    const float* query_vector,     // [d]
    const float* lattice_points,   // [lut_size, d]
    float* lut,                    // [lut_size]
    int lut_size,
    int d
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i >= lut_size) return;
    
    // Compute inner product with query vector
    float inner_product = 0.0f;
    for (int k = 0; k < d; k++) {
        inner_product += query_vector[k] * lattice_points[i * d + k];
    }
    
    lut[i] = inner_product;
}

// CUDA kernel for LUT lookup
__global__ void lut_lookup_kernel(
    const int8_t* indices_x,       // [batch_size]
    const int8_t* indices_y,       // [batch_size]
    const int32_t* lut,            // [lut_size, lut_size]
    int32_t* results,              // [batch_size]
    int batch_size,
    int lut_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    // Look up in LUT
    int idx_x = indices_x[idx];
    int idx_y = indices_y[idx];
    
    if (idx_x >= 0 && idx_x < lut_size && idx_y >= 0 && idx_y < lut_size) {
        results[idx] = lut[idx_x * lut_size + idx_y];
    } else {
        results[idx] = 0;
    }
}

// CUDA kernel for one-sided LUT lookup
__global__ void one_sided_lut_lookup_kernel(
    const int8_t* indices,         // [batch_size]
    const float* lut,              // [lut_size]
    float* results,                // [batch_size]
    int batch_size,
    int lut_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    // Look up in LUT
    int idx_val = indices[idx];
    
    if (idx_val >= 0 && idx_val < lut_size) {
        results[idx] = lut[idx_val];
    } else {
        results[idx] = 0.0f;
    }
}

// CUDA kernel for batch LUT lookup
__global__ void batch_lut_lookup_kernel(
    const int8_t* indices_batch_x, // [batch_size, M]
    const int8_t* indices_batch_y, // [batch_size, M]
    const int32_t* lut,            // [lut_size, lut_size]
    int64_t* results,              // [batch_size]
    int batch_size,
    int M,
    int lut_size,
    int q
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    int64_t result = 0;
    
    // Compute MAC using LUT
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            int idx_x = indices_batch_x[idx * M + i];
            int idx_y = indices_batch_y[idx * M + j];
            
            if (idx_x >= 0 && idx_x < lut_size && idx_y >= 0 && idx_y < lut_size) {
                int32_t lut_value = lut[idx_x * lut_size + idx_y];
                
                // Scale by q^(i+j)
                int64_t scale_factor = 1;
                for (int s = 0; s < i + j; s++) {
                    scale_factor *= q;
                }
                
                result += lut_value * scale_factor;
            }
        }
    }
    
    results[idx] = result;
}

// CUDA kernel for LUT cache management
__global__ void clear_lut_cache_kernel(
    int32_t* lut_cache,            // [cache_size]
    int cache_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= cache_size) return;
    
    lut_cache[idx] = 0;
}

// Host function to launch build LUT kernel
torch::Tensor cuda_build_lut(
    const torch::Tensor& lattice_points,
    int lut_size,
    int d
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(lattice_points.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Create output tensor
    auto lut = torch::zeros({lut_size, lut_size}, 
        torch::TensorOptions().dtype(torch::kInt32).device(lattice_points.device()));
    
    // Launch kernel
    dim3 threads_per_block(16, 16);
    dim3 num_blocks(
        (lut_size + threads_per_block.x - 1) / threads_per_block.x,
        (lut_size + threads_per_block.y - 1) / threads_per_block.y
    );
    
    build_lut_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        lattice_points.data_ptr<float>(),
        lut.data_ptr<int32_t>(),
        lut_size,
        d
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return lut;
}

// Host function to launch build one-sided LUT kernel
torch::Tensor cuda_build_one_sided_lut(
    const torch::Tensor& query_vector,
    const torch::Tensor& lattice_points,
    int lut_size,
    int d
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(query_vector.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Create output tensor
    auto lut = torch::zeros({lut_size}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(query_vector.device()));
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (lut_size + threads_per_block - 1) / threads_per_block;
    
    build_one_sided_lut_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        query_vector.data_ptr<float>(),
        lattice_points.data_ptr<float>(),
        lut.data_ptr<float>(),
        lut_size,
        d
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return lut;
}

// Host function to launch LUT lookup kernel
torch::Tensor cuda_lut_lookup(
    const torch::Tensor& indices_x,
    const torch::Tensor& indices_y,
    const torch::Tensor& lut
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(indices_x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Get dimensions
    int batch_size = indices_x.size(0);
    int lut_size = lut.size(0);
    
    // Create output tensor
    auto results = torch::zeros({batch_size}, 
        torch::TensorOptions().dtype(torch::kInt32).device(indices_x.device()));
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    lut_lookup_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        indices_x.data_ptr<int8_t>(),
        indices_y.data_ptr<int8_t>(),
        lut.data_ptr<int32_t>(),
        results.data_ptr<int32_t>(),
        batch_size,
        lut_size
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return results;
}

// Host function to launch one-sided LUT lookup kernel
torch::Tensor cuda_one_sided_lut_lookup(
    const torch::Tensor& indices,
    const torch::Tensor& lut
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(indices.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Get dimensions
    int batch_size = indices.size(0);
    int lut_size = lut.size(0);
    
    // Create output tensor
    auto results = torch::zeros({batch_size}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(indices.device()));
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    one_sided_lut_lookup_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        indices.data_ptr<int8_t>(),
        lut.data_ptr<float>(),
        results.data_ptr<float>(),
        batch_size,
        lut_size
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return results;
}

// Host function to launch batch LUT lookup kernel
torch::Tensor cuda_batch_lut_lookup(
    const torch::Tensor& indices_batch_x,
    const torch::Tensor& indices_batch_y,
    const torch::Tensor& lut,
    int M,
    int q
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(indices_batch_x.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Get dimensions
    int batch_size = indices_batch_x.size(0);
    int lut_size = lut.size(0);
    
    // Create output tensor
    auto results = torch::zeros({batch_size}, 
        torch::TensorOptions().dtype(torch::kInt64).device(indices_batch_x.device()));
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    batch_lut_lookup_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        indices_batch_x.data_ptr<int8_t>(),
        indices_batch_y.data_ptr<int8_t>(),
        lut.data_ptr<int32_t>(),
        results.data_ptr<int64_t>(),
        batch_size,
        M,
        lut_size,
        q
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return results;
}

// Host function to launch clear LUT cache kernel
void cuda_clear_lut_cache(
    torch::Tensor& lut_cache
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(lut_cache.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Get dimensions
    int cache_size = lut_cache.numel();
    
    // Launch kernel
    int threads_per_block = 256;
    int num_blocks = (cache_size + threads_per_block - 1) / threads_per_block;
    
    clear_lut_cache_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        lut_cache.data_ptr<int32_t>(),
        cache_size
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
}
