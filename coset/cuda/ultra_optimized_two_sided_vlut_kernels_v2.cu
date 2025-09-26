#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>
#include <c10/cuda/CUDAGuard.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Ultra-optimized encoding-to-index conversion with warp-level primitives
__device__ __forceinline__ int ultra_optimized_encoding_to_index(
    const int32_t* encoding, 
    int d, 
    int q
) {
    // Pre-computed powers of q for E8 lattice (q=3)
    const int powers[8] = {1, 3, 9, 27, 81, 243, 729, 2187};
    
    int index = 0;
    
    // Use warp-level primitives for vectorized operations
    if (d >= 8) {
        int val8 = encoding[7];
        index += __shfl_sync(0xffffffff, val8 * powers[7], 0);
    }
    if (d >= 7) {
        int val7 = encoding[6];
        index += __shfl_sync(0xffffffff, val7 * powers[6], 0);
    }
    if (d >= 6) {
        int val6 = encoding[5];
        index += __shfl_sync(0xffffffff, val6 * powers[5], 0);
    }
    if (d >= 5) {
        int val5 = encoding[4];
        index += __shfl_sync(0xffffffff, val5 * powers[4], 0);
    }
    if (d >= 4) {
        int val4 = encoding[3];
        index += __shfl_sync(0xffffffff, val4 * powers[3], 0);
    }
    if (d >= 3) {
        int val3 = encoding[2];
        index += __shfl_sync(0xffffffff, val3 * powers[2], 0);
    }
    if (d >= 2) {
        int val2 = encoding[1];
        index += __shfl_sync(0xffffffff, val2 * powers[1], 0);
    }
    if (d >= 1) {
        int val1 = encoding[0];
        index += __shfl_sync(0xffffffff, val1 * powers[0], 0);
    }
    
    return index;
}

// Ultra-optimized two-sided vLUT construction with shared memory
__global__ void ultra_optimized_build_two_sided_vlut_kernel(
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
    // Shared memory for input and query vectors
    extern __shared__ float shared_vectors[];
    float* shared_inputs = shared_vectors;
    float* shared_queries = shared_vectors + num_inputs * d;
    
    // Load vectors into shared memory
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    // Load input vectors
    for (int i = tid; i < num_inputs * d; i += block_size) {
        shared_inputs[i] = input_vectors[i];
    }
    
    // Load query vectors
    for (int i = tid; i < num_queries * d; i += block_size) {
        shared_queries[i] = query_vectors[i];
    }
    __syncthreads();
    
    // Use 2D grid for input × query parallelism
    int input_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (input_idx >= num_inputs || query_idx >= num_queries) return;
    
    // Get input and query vectors from shared memory
    const float* input_vec = &shared_inputs[input_idx * d];
    const float* query_vec = &shared_queries[query_idx * d];
    
    // Compute dot product for normalization using warp-level reduction
    float dot_product = 0.0f;
    for (int i = 0; i < d; i++) {
        dot_product += input_vec[i] * query_vec[i];
    }
    
    // Use warp-level reduction for dot product
    dot_product = __shfl_down_sync(0xffffffff, dot_product, 16);
    dot_product = __shfl_down_sync(0xffffffff, dot_product, 8);
    dot_product = __shfl_down_sync(0xffffffff, dot_product, 4);
    dot_product = __shfl_down_sync(0xffffffff, dot_product, 2);
    dot_product = __shfl_down_sync(0xffffffff, dot_product, 1);
    
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
        
        // Use warp-level reduction for lattice dot product
        lattice_dot = __shfl_down_sync(0xffffffff, lattice_dot, 16);
        lattice_dot = __shfl_down_sync(0xffffffff, lattice_dot, 8);
        lattice_dot = __shfl_down_sync(0xffffffff, lattice_dot, 4);
        lattice_dot = __shfl_down_sync(0xffffffff, lattice_dot, 2);
        lattice_dot = __shfl_down_sync(0xffffffff, lattice_dot, 1);
        
        // Store vLUT value
        int vlut_offset = (input_idx * num_queries + query_idx) * lut_size + lut_idx;
        vlut[vlut_offset] = lattice_dot;
    }
}

// Ultra-optimized two-sided vLUT MAC with shared memory
__global__ void ultra_optimized_two_sided_vlut_mac_kernel(
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
    // Shared memory for vLUT caching
    extern __shared__ float shared_vlut[];
    
    // Load vLUT into shared memory
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    for (int i = tid; i < num_queries * lut_size; i += block_size) {
        shared_vlut[i] = vlut[i];
    }
    __syncthreads();
    
    // Use 2D grid for batch × query parallelism
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || query_idx >= num_queries) return;
    
    // Ultra-optimized encoding-to-index conversion for both input and query
    int input_encoding_idx = ultra_optimized_encoding_to_index(
        &input_encodings[batch_idx * d], d, q
    );
    int query_encoding_idx = ultra_optimized_encoding_to_index(
        &query_encodings[query_idx * d], d, q
    );
    
    // Lookup in shared memory vLUT
    float result = 0.0f;
    if (input_encoding_idx >= 0 && input_encoding_idx < lut_size &&
        query_encoding_idx >= 0 && query_encoding_idx < lut_size) {
        
        int vlut_offset = query_idx * lut_size + input_encoding_idx;
        result = shared_vlut[vlut_offset];
    }
    
    // Use warp-level reduction for result
    result = __shfl_down_sync(0xffffffff, result, 16);
    result = __shfl_down_sync(0xffffffff, result, 8);
    result = __shfl_down_sync(0xffffffff, result, 4);
    result = __shfl_down_sync(0xffffffff, result, 2);
    result = __shfl_down_sync(0xffffffff, result, 1);
    
    // Store result
    results[batch_idx * num_queries + query_idx] = result;
}

// Ultra-optimized batch two-sided vLUT operations with shared memory
__global__ void ultra_optimized_batch_two_sided_vlut_kernel(
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
    // Shared memory for vLUT caching
    extern __shared__ float shared_vluts[];
    
    // Load vLUTs into shared memory
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int shared_size = num_queries * input_dim * lut_size;
    
    for (int i = tid; i < shared_size; i += block_size) {
        shared_vluts[i] = vluts[i];
    }
    __syncthreads();
    
    // Use 2D grid for batch × query parallelism
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int query_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || query_idx >= num_queries) return;
    
    float result = 0.0f;
    
    // Process all input dimensions with warp-level optimization
    for (int in_idx = 0; in_idx < input_dim; in_idx++) {
        // Ultra-optimized encoding-to-index conversion
        int input_encoding_idx = ultra_optimized_encoding_to_index(
            &input_encodings[batch_idx * input_dim * d + in_idx * d], d, q
        );
        int query_encoding_idx = ultra_optimized_encoding_to_index(
            &query_encodings[query_idx * d], d, q
        );
        
        // Lookup in shared memory vLUT
        if (input_encoding_idx >= 0 && input_encoding_idx < lut_size &&
            query_encoding_idx >= 0 && query_encoding_idx < lut_size) {
            
            int vlut_offset = (query_idx * input_dim + in_idx) * lut_size + input_encoding_idx;
            result += shared_vluts[vlut_offset];
        }
    }
    
    // Use warp-level reduction for accumulation
    result = __shfl_down_sync(0xffffffff, result, 16);
    result = __shfl_down_sync(0xffffffff, result, 8);
    result = __shfl_down_sync(0xffffffff, result, 4);
    result = __shfl_down_sync(0xffffffff, result, 2);
    result = __shfl_down_sync(0xffffffff, result, 1);
    
    // Store result
    results[batch_idx * num_queries + query_idx] = result;
}

// Ultra-optimized two-sided vLUT matrix multiplication with shared memory
__global__ void ultra_optimized_two_sided_vlut_matrix_multiply_kernel(
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
    // Shared memory for vLUT caching
    extern __shared__ float shared_vluts[];
    
    // Load vLUTs into shared memory
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int shared_size = output_dim * input_dim * lut_size;
    
    for (int i = tid; i < shared_size; i += block_size) {
        shared_vluts[i] = vluts[i];
    }
    __syncthreads();
    
    // Use 2D grid for batch × output parallelism
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (batch_idx >= batch_size || out_idx >= output_dim) return;
    
    float result = 0.0f;
    
    // Process all input dimensions with warp-level optimization
    for (int in_idx = 0; in_idx < input_dim; in_idx++) {
        // Ultra-optimized encoding-to-index conversion
        int input_encoding_idx = ultra_optimized_encoding_to_index(
            &input_encodings[batch_idx * input_dim * d + in_idx * d], d, q
        );
        int weight_encoding_idx = ultra_optimized_encoding_to_index(
            &weight_encodings[out_idx * input_dim * d + in_idx * d], d, q
        );
        
        // Lookup in shared memory vLUT
        if (input_encoding_idx >= 0 && input_encoding_idx < lut_size &&
            weight_encoding_idx >= 0 && weight_encoding_idx < lut_size) {
            
            int vlut_offset = (out_idx * input_dim + in_idx) * lut_size + input_encoding_idx;
            result += shared_vluts[vlut_offset];
        }
    }
    
    // Use warp-level reduction for accumulation
    result = __shfl_down_sync(0xffffffff, result, 16);
    result = __shfl_down_sync(0xffffffff, result, 8);
    result = __shfl_down_sync(0xffffffff, result, 4);
    result = __shfl_down_sync(0xffffffff, result, 2);
    result = __shfl_down_sync(0xffffffff, result, 1);
    
    // Store result
    results[batch_idx * output_dim + out_idx] = result;
}

// Host function to launch ultra-optimized two-sided vLUT construction
torch::Tensor cuda_ultra_optimized_build_two_sided_vlut(
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
    
    // Use larger thread blocks with shared memory
    dim3 threads_per_block(32, 32);  // 1024 threads per block
    dim3 num_blocks(
        (num_inputs + threads_per_block.x - 1) / threads_per_block.x,
        (num_queries + threads_per_block.y - 1) / threads_per_block.y
    );
    
    int shared_mem_size = (num_inputs + num_queries) * d * sizeof(float);
    
    ultra_optimized_build_two_sided_vlut_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
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

// Host function to launch ultra-optimized two-sided vLUT MAC
torch::Tensor cuda_ultra_optimized_two_sided_vlut_mac(
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
    
    // Use larger thread blocks with shared memory
    dim3 threads_per_block(32, 32);  // 1024 threads per block
    dim3 num_blocks(
        (batch_size + threads_per_block.x - 1) / threads_per_block.x,
        (num_queries + threads_per_block.y - 1) / threads_per_block.y
    );
    
    int shared_mem_size = num_queries * lut_size * sizeof(float);
    
    ultra_optimized_two_sided_vlut_mac_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
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

// Host function to launch ultra-optimized batch two-sided vLUT operations
torch::Tensor cuda_ultra_optimized_batch_two_sided_vlut(
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
    
    // Use larger thread blocks with shared memory
    dim3 threads_per_block(32, 32);  // 1024 threads per block
    dim3 num_blocks(
        (batch_size + threads_per_block.x - 1) / threads_per_block.x,
        (num_queries + threads_per_block.y - 1) / threads_per_block.y
    );
    
    int shared_mem_size = num_queries * input_dim * lut_size * sizeof(float);
    
    ultra_optimized_batch_two_sided_vlut_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
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

// Host function to launch ultra-optimized two-sided vLUT matrix multiply
torch::Tensor cuda_ultra_optimized_two_sided_vlut_matrix_multiply(
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
    
    // Use larger thread blocks with shared memory
    dim3 threads_per_block(32, 32);  // 1024 threads per block
    dim3 num_blocks(
        (batch_size + threads_per_block.x - 1) / threads_per_block.x,
        (output_dim + threads_per_block.y - 1) / threads_per_block.y
    );
    
    int shared_mem_size = output_dim * input_dim * lut_size * sizeof(float);
    
    ultra_optimized_two_sided_vlut_matrix_multiply_kernel<<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
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
    m.def("cuda_ultra_optimized_build_two_sided_vlut", &cuda_ultra_optimized_build_two_sided_vlut, "Ultra-optimized two-sided vLUT construction");
    m.def("cuda_ultra_optimized_two_sided_vlut_mac", &cuda_ultra_optimized_two_sided_vlut_mac, "Ultra-optimized two-sided vLUT MAC");
    m.def("cuda_ultra_optimized_batch_two_sided_vlut", &cuda_ultra_optimized_batch_two_sided_vlut, "Ultra-optimized batch two-sided vLUT operations");
    m.def("cuda_ultra_optimized_two_sided_vlut_matrix_multiply", &cuda_ultra_optimized_two_sided_vlut_matrix_multiply, "Ultra-optimized two-sided vLUT matrix multiply");
}
