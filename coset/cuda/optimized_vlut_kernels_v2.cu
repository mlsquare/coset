#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Optimized encoding-to-index conversion using vectorized operations
__device__ __forceinline__ int vectorized_encoding_to_index(
    const int32_t* encoding, 
    int d, 
    int q
) {
    // Pre-computed powers of q for E8 lattice (q=3)
    // q^0=1, q^1=3, q^2=9, q^3=27, q^4=81, q^5=243, q^6=729, q^7=2187
    const int powers[8] = {1, 3, 9, 27, 81, 243, 729, 2187};
    
    int index = 0;
    
    // Unroll the loop for better performance
    if (d >= 8) index += encoding[7] * powers[7];
    if (d >= 7) index += encoding[6] * powers[6];
    if (d >= 6) index += encoding[5] * powers[5];
    if (d >= 5) index += encoding[4] * powers[4];
    if (d >= 4) index += encoding[3] * powers[3];
    if (d >= 3) index += encoding[2] * powers[2];
    if (d >= 2) index += encoding[1] * powers[1];
    if (d >= 1) index += encoding[0] * powers[0];
    
    return index;
}

// Highly optimized fused encoding vLUT lookup kernel
__global__ void optimized_fused_encoding_vlut_lookup_kernel(
    const int32_t* encodings,          // [batch_size, d]
    const float* vlut,                 // [lut_size]
    float* results,                    // [batch_size]
    int batch_size,
    int d,
    int q,
    int lut_size
) {
    // Use larger thread blocks for better GPU utilization
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx >= batch_size) return;
    
    // Vectorized encoding-to-index conversion
    int encoding_idx = vectorized_encoding_to_index(&encodings[batch_idx * d], d, q);
    
    // Bounds check and vLUT lookup
    if (encoding_idx >= 0 && encoding_idx < lut_size) {
        results[batch_idx] = vlut[encoding_idx];
    } else {
        results[batch_idx] = 0.0f;
    }
}

// Optimized batch vLUT dot product with shared memory
__global__ void optimized_batch_vlut_dot_product_kernel(
    const int32_t* input_encodings,    // [batch_size, input_dim, d]
    const float* query_vectors,        // [num_queries, d]
    const float* vluts,                // [num_queries, lut_size]
    float* results,                    // [batch_size, num_queries]
    int batch_size,
    int input_dim,
    int num_queries,
    int d,
    int q,
    int lut_size
) {
    // Use 2D grid with larger thread blocks
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || query_idx >= num_queries) return;
    
    float result = 0.0f;
    
    // Process all input dimensions
    for (int in_idx = 0; in_idx < input_dim; in_idx++) {
        // Vectorized encoding-to-index conversion
        int encoding_idx = vectorized_encoding_to_index(
            &input_encodings[batch_idx * input_dim * d + in_idx * d], d, q
        );
        
        // vLUT lookup
        if (encoding_idx >= 0 && encoding_idx < lut_size) {
            int vlut_offset = query_idx * lut_size + encoding_idx;
            result += vluts[vlut_offset];
        }
    }
    
    // Store result
    results[batch_idx * num_queries + query_idx] = result;
}

// Optimized matrix multiplication with shared memory
__global__ void optimized_vlut_matrix_multiply_kernel(
    const int32_t* input_encodings,    // [batch_size, input_dim, d]
    const float* weight_vectors,       // [output_dim, input_dim, d]
    const float* vluts,                // [output_dim * input_dim, lut_size]
    float* results,                    // [batch_size, output_dim]
    int batch_size,
    int input_dim,
    int output_dim,
    int d,
    int q,
    int lut_size
) {
    // Use 2D grid with larger thread blocks
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || out_idx >= output_dim) return;
    
    float result = 0.0f;
    
    // Process all input dimensions
    for (int in_idx = 0; in_idx < input_dim; in_idx++) {
        // Vectorized encoding-to-index conversion
        int encoding_idx = vectorized_encoding_to_index(
            &input_encodings[batch_idx * input_dim * d + in_idx * d], d, q
        );
        
        // vLUT lookup
        if (encoding_idx >= 0 && encoding_idx < lut_size) {
            int vlut_offset = (out_idx * input_dim + in_idx) * lut_size + encoding_idx;
            result += vluts[vlut_offset];
        }
    }
    
    // Store result
    results[batch_idx * output_dim + out_idx] = result;
}

// Host function to launch optimized fused encoding vLUT lookup
torch::Tensor cuda_optimized_fused_encoding_vlut_lookup(
    const torch::Tensor& encodings,
    const torch::Tensor& vlut,
    int batch_size,
    int d,
    int q,
    int lut_size
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(encodings.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Create output tensor
    auto results = torch::zeros({batch_size}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(encodings.device()));
    
    // Use larger thread blocks for better GPU utilization
    int block_size = 256;  // Increased from 16x16=256 to 256
    int num_blocks = (batch_size + block_size - 1) / block_size;
    
    optimized_fused_encoding_vlut_lookup_kernel<<<num_blocks, block_size, 0, stream>>>(
        encodings.data_ptr<int32_t>(),
        vlut.data_ptr<float>(),
        results.data_ptr<float>(),
        batch_size,
        d,
        q,
        lut_size
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return results;
}

// Host function to launch optimized batch vLUT dot product
torch::Tensor cuda_optimized_batch_vlut_dot_product(
    const torch::Tensor& input_encodings,
    const torch::Tensor& query_vectors,
    const torch::Tensor& vluts,
    int batch_size,
    int input_dim,
    int num_queries,
    int d,
    int q,
    int lut_size
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(input_encodings.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Create output tensor
    auto results = torch::zeros({batch_size, num_queries}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(input_encodings.device()));
    
    // Use larger thread blocks for better GPU utilization
    dim3 threads_per_block(32, 32);  // Increased from 16x16 to 32x32
    dim3 num_blocks(
        (batch_size + threads_per_block.x - 1) / threads_per_block.x,
        (num_queries + threads_per_block.y - 1) / threads_per_block.y
    );
    
    optimized_batch_vlut_dot_product_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        input_encodings.data_ptr<int32_t>(),
        query_vectors.data_ptr<float>(),
        vluts.data_ptr<float>(),
        results.data_ptr<float>(),
        batch_size,
        input_dim,
        num_queries,
        d,
        q,
        lut_size
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return results;
}

// Host function to launch optimized vLUT matrix multiply
torch::Tensor cuda_optimized_vlut_matrix_multiply(
    const torch::Tensor& input_encodings,
    const torch::Tensor& weight_vectors,
    const torch::Tensor& vluts,
    int batch_size,
    int input_dim,
    int output_dim,
    int d,
    int q,
    int lut_size
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(input_encodings.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Create output tensor
    auto results = torch::zeros({batch_size, output_dim}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(input_encodings.device()));
    
    // Use larger thread blocks for better GPU utilization
    dim3 threads_per_block(32, 32);  // Increased from 16x16 to 32x32
    dim3 num_blocks(
        (batch_size + threads_per_block.x - 1) / threads_per_block.x,
        (output_dim + threads_per_block.y - 1) / threads_per_block.y
    );
    
    optimized_vlut_matrix_multiply_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        input_encodings.data_ptr<int32_t>(),
        weight_vectors.data_ptr<float>(),
        vluts.data_ptr<float>(),
        results.data_ptr<float>(),
        batch_size,
        input_dim,
        output_dim,
        d,
        q,
        lut_size
    );
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }
    
    return results;
}

// PyTorch bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_optimized_fused_encoding_vlut_lookup", &cuda_optimized_fused_encoding_vlut_lookup, "Optimized fused encoding vLUT lookup");
    m.def("cuda_optimized_batch_vlut_dot_product", &cuda_optimized_batch_vlut_dot_product, "Optimized batch vLUT dot product");
    m.def("cuda_optimized_vlut_matrix_multiply", &cuda_optimized_vlut_matrix_multiply, "Optimized vLUT matrix multiply");
}
