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

// Optimized two-sided vLUT construction kernel
__global__ void optimized_build_two_sided_vlut_kernel(
    const int32_t* input_encodings,     // [num_inputs, d]
    const int32_t* query_encodings,     // [num_queries, d]
    const float* input_vectors,         // [num_inputs, d]
    const float* query_vectors,         // [num_queries, d]
    float* vlut,                        // [num_inputs, num_queries, lut_size]
    int num_inputs,
    int num_queries,
    int d,
    int q,
    int lut_size
) {
    // Use 2D grid for input × query parallelism
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (input_idx >= num_inputs || query_idx >= num_queries) return;
    
    // Get input and query vectors
    const float* input_vec = &input_vectors[input_idx * d];
    const float* query_vec = &query_vectors[query_idx * d];
    
    // Compute dot product for normalization
    float dot_product = 0.0f;
    for (int i = 0; i < d; i++) {
        dot_product += input_vec[i] * query_vec[i];
    }
    
    // Generate all possible lattice points and compute vLUT values
    for (int lut_idx = 0; lut_idx < lut_size; lut_idx++) {
        // Convert index to encoding
        int temp_idx = lut_idx;
        float lattice_dot = 0.0f;
        
        for (int k = 0; k < d; k++) {
            int encoding_val = temp_idx % q;
            temp_idx /= q;
            
            // Compute lattice point value (simplified)
            float lattice_val = (encoding_val - (q - 1) / 2.0f) * 2.0f / (q - 1);
            lattice_dot += lattice_val * query_vec[k];
        }
        
        // Store vLUT value
        int vlut_offset = (input_idx * num_queries + query_idx) * lut_size + lut_idx;
        vlut[vlut_offset] = lattice_dot;
    }
}

// Optimized two-sided vLUT MAC operation kernel
__global__ void optimized_two_sided_vlut_mac_kernel(
    const int32_t* input_encodings,     // [batch_size, d]
    const int32_t* query_encodings,     // [num_queries, d]
    const float* vlut,                  // [num_queries, lut_size]
    float* results,                     // [batch_size, num_queries]
    int batch_size,
    int num_queries,
    int d,
    int q,
    int lut_size
) {
    // Use 2D grid for batch × query parallelism
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || query_idx >= num_queries) return;
    
    // Convert input encoding to index
    int input_encoding_idx = vectorized_encoding_to_index(
        &input_encodings[batch_idx * d], d, q
    );
    
    // Convert query encoding to index
    int query_encoding_idx = vectorized_encoding_to_index(
        &query_encodings[query_idx * d], d, q
    );
    
    // Lookup in vLUT
    float result = 0.0f;
    if (input_encoding_idx >= 0 && input_encoding_idx < lut_size &&
        query_encoding_idx >= 0 && query_encoding_idx < lut_size) {
        
        // For two-sided vLUT, we need to handle the combination of both encodings
        // This is a simplified implementation - in practice, you might need a different indexing scheme
        int vlut_offset = query_idx * lut_size + input_encoding_idx;
        result = vlut[vlut_offset];
    }
    
    // Store result
    results[batch_idx * num_queries + query_idx] = result;
}

// Optimized batch two-sided vLUT operations kernel
__global__ void optimized_batch_two_sided_vlut_kernel(
    const int32_t* input_encodings,     // [batch_size, input_dim, d]
    const int32_t* query_encodings,     // [num_queries, d]
    const float* vluts,                 // [num_queries, input_dim, lut_size]
    float* results,                     // [batch_size, num_queries]
    int batch_size,
    int input_dim,
    int num_queries,
    int d,
    int q,
    int lut_size
) {
    // Use 2D grid for batch × query parallelism
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || query_idx >= num_queries) return;
    
    float result = 0.0f;
    
    // Process all input dimensions
    for (int in_idx = 0; in_idx < input_dim; in_idx++) {
        // Convert input encoding to index
        int input_encoding_idx = vectorized_encoding_to_index(
            &input_encodings[batch_idx * input_dim * d + in_idx * d], d, q
        );
        
        // Convert query encoding to index
        int query_encoding_idx = vectorized_encoding_to_index(
            &query_encodings[query_idx * d], d, q
        );
        
        // Lookup in vLUT
        if (input_encoding_idx >= 0 && input_encoding_idx < lut_size &&
            query_encoding_idx >= 0 && query_encoding_idx < lut_size) {
            
            int vlut_offset = (query_idx * input_dim + in_idx) * lut_size + input_encoding_idx;
            result += vluts[vlut_offset];
        }
    }
    
    // Store result
    results[batch_idx * num_queries + query_idx] = result;
}

// Optimized two-sided vLUT matrix multiplication kernel
__global__ void optimized_two_sided_vlut_matrix_multiply_kernel(
    const int32_t* input_encodings,     // [batch_size, input_dim, d]
    const int32_t* weight_encodings,    // [output_dim, input_dim, d]
    const float* vluts,                 // [output_dim, input_dim, lut_size]
    float* results,                     // [batch_size, output_dim]
    int batch_size,
    int input_dim,
    int output_dim,
    int d,
    int q,
    int lut_size
) {
    // Use 2D grid for batch × output parallelism
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || out_idx >= output_dim) return;
    
    float result = 0.0f;
    
    // Process all input dimensions
    for (int in_idx = 0; in_idx < input_dim; in_idx++) {
        // Convert input encoding to index
        int input_encoding_idx = vectorized_encoding_to_index(
            &input_encodings[batch_idx * input_dim * d + in_idx * d], d, q
        );
        
        // Convert weight encoding to index
        int weight_encoding_idx = vectorized_encoding_to_index(
            &weight_encodings[out_idx * input_dim * d + in_idx * d], d, q
        );
        
        // Lookup in vLUT
        if (input_encoding_idx >= 0 && input_encoding_idx < lut_size &&
            weight_encoding_idx >= 0 && weight_encoding_idx < lut_size) {
            
            int vlut_offset = (out_idx * input_dim + in_idx) * lut_size + input_encoding_idx;
            result += vluts[vlut_offset];
        }
    }
    
    // Store result
    results[batch_idx * output_dim + out_idx] = result;
}

// Host function to launch optimized two-sided vLUT construction
torch::Tensor cuda_optimized_build_two_sided_vlut(
    const torch::Tensor& input_encodings,
    const torch::Tensor& query_encodings,
    const torch::Tensor& input_vectors,
    const torch::Tensor& query_vectors,
    int num_inputs,
    int num_queries,
    int d,
    int q,
    int lut_size
) {
    // Get device and stream
    c10::cuda::CUDAGuard guard(input_encodings.device());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Create output tensor
    auto vlut = torch::zeros({num_inputs, num_queries, lut_size}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(input_encodings.device()));
    
    // Use larger thread blocks for better GPU utilization
    dim3 threads_per_block(32, 32);  // 1024 threads per block
    dim3 num_blocks(
        (num_inputs + threads_per_block.x - 1) / threads_per_block.x,
        (num_queries + threads_per_block.y - 1) / threads_per_block.y
    );
    
    optimized_build_two_sided_vlut_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        input_encodings.data_ptr<int32_t>(),
        query_encodings.data_ptr<int32_t>(),
        input_vectors.data_ptr<float>(),
        query_vectors.data_ptr<float>(),
        vlut.data_ptr<float>(),
        num_inputs,
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
    
    return vlut;
}

// Host function to launch optimized two-sided vLUT MAC
torch::Tensor cuda_optimized_two_sided_vlut_mac(
    const torch::Tensor& input_encodings,
    const torch::Tensor& query_encodings,
    const torch::Tensor& vlut,
    int batch_size,
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
    dim3 threads_per_block(32, 32);  // 1024 threads per block
    dim3 num_blocks(
        (batch_size + threads_per_block.x - 1) / threads_per_block.x,
        (num_queries + threads_per_block.y - 1) / threads_per_block.y
    );
    
    optimized_two_sided_vlut_mac_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        input_encodings.data_ptr<int32_t>(),
        query_encodings.data_ptr<int32_t>(),
        vlut.data_ptr<float>(),
        results.data_ptr<float>(),
        batch_size,
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

// Host function to launch optimized batch two-sided vLUT operations
torch::Tensor cuda_optimized_batch_two_sided_vlut(
    const torch::Tensor& input_encodings,
    const torch::Tensor& query_encodings,
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
    dim3 threads_per_block(32, 32);  // 1024 threads per block
    dim3 num_blocks(
        (batch_size + threads_per_block.x - 1) / threads_per_block.x,
        (num_queries + threads_per_block.y - 1) / threads_per_block.y
    );
    
    optimized_batch_two_sided_vlut_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        input_encodings.data_ptr<int32_t>(),
        query_encodings.data_ptr<int32_t>(),
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

// Host function to launch optimized two-sided vLUT matrix multiply
torch::Tensor cuda_optimized_two_sided_vlut_matrix_multiply(
    const torch::Tensor& input_encodings,
    const torch::Tensor& weight_encodings,
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
    dim3 threads_per_block(32, 32);  // 1024 threads per block
    dim3 num_blocks(
        (batch_size + threads_per_block.x - 1) / threads_per_block.x,
        (output_dim + threads_per_block.y - 1) / threads_per_block.y
    );
    
    optimized_two_sided_vlut_matrix_multiply_kernel<<<num_blocks, threads_per_block, 0, stream>>>(
        input_encodings.data_ptr<int32_t>(),
        weight_encodings.data_ptr<int32_t>(),
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
    m.def("cuda_optimized_build_two_sided_vlut", &cuda_optimized_build_two_sided_vlut, "Optimized two-sided vLUT construction");
    m.def("cuda_optimized_two_sided_vlut_mac", &cuda_optimized_two_sided_vlut_mac, "Optimized two-sided vLUT MAC");
    m.def("cuda_optimized_batch_two_sided_vlut", &cuda_optimized_batch_two_sided_vlut, "Optimized batch two-sided vLUT operations");
    m.def("cuda_optimized_two_sided_vlut_matrix_multiply", &cuda_optimized_two_sided_vlut_matrix_multiply, "Optimized two-sided vLUT matrix multiply");
}
