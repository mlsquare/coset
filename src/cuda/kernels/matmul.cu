#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Ultra-fast quantized matrix multiplication kernel
__global__ void quantized_matmul_kernel(
    const int* input_indices,
    const int* weight_indices,
    const float* lookup_table,
    const float* bias,
    float* output,
    int batch_size,
    int out_features,
    int num_blocks,
    int lattice_dim,
    int lookup_table_size
) {
    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x;
    int thread_id = threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= out_features) return;
    
    // Shared memory for lookup table (if it fits)
    extern __shared__ float shared_lut[];
    
    // Load lookup table into shared memory
    int lut_size = lookup_table_size * lookup_table_size;
    for (int i = thread_id; i < lut_size; i += blockDim.x) {
        shared_lut[i] = lookup_table[i];
    }
    __syncthreads();
    
    float sum = 0.0f;
    int total_elements = num_blocks * lattice_dim;
    
    // Process elements in parallel
    for (int elem_idx = thread_id; elem_idx < total_elements; elem_idx += blockDim.x) {
        int block_idx = elem_idx / lattice_dim;
        int dim_idx = elem_idx % lattice_dim;
        
        // Get input and weight indices
        int input_idx = input_indices[batch_idx * total_elements + elem_idx];
        int weight_idx = weight_indices[out_idx * total_elements + elem_idx];
        
        // Clamp indices to valid range
        input_idx = max(0, min(input_idx, lookup_table_size - 1));
        weight_idx = max(0, min(weight_idx, lookup_table_size - 1));
        
        // Lookup value from shared memory
        float lookup_value = shared_lut[input_idx * lookup_table_size + weight_idx];
        
        // Atomic add to avoid race conditions
        atomicAdd(&sum, lookup_value);
    }
    __syncthreads();
    
    // Write result
    if (thread_id == 0) {
        float result = sum;
        if (bias != nullptr) {
            result += bias[out_idx];
        }
        output[batch_idx * out_features + out_idx] = result;
    }
}

// Optimized kernel for smaller lookup tables
__global__ void quantized_matmul_small_kernel(
    const int* input_indices,
    const int* weight_indices,
    const float* lookup_table,
    const float* bias,
    float* output,
    int batch_size,
    int out_features,
    int num_blocks,
    int lattice_dim,
    int lookup_table_size
) {
    int batch_idx = blockIdx.y;
    int out_idx = blockIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= out_features) return;
    
    float sum = 0.0f;
    int total_elements = num_blocks * lattice_dim;
    
    // Unrolled loop for better performance
    #pragma unroll
    for (int i = 0; i < total_elements; i++) {
        // Get input and weight indices
        int input_idx = input_indices[batch_idx * total_elements + i];
        int weight_idx = weight_indices[out_idx * total_elements + i];
        
        // Clamp indices
        input_idx = max(0, min(input_idx, lookup_table_size - 1));
        weight_idx = max(0, min(weight_idx, lookup_table_size - 1));
        
        // Direct lookup
        float lookup_value = lookup_table[input_idx * lookup_table_size + weight_idx];
        sum += lookup_value;
    }
    
    // Add bias
    if (bias != nullptr) {
        sum += bias[out_idx];
    }
    
    output[batch_idx * out_features + out_idx] = sum;
}

torch::Tensor quantized_matmul_cuda(
    torch::Tensor input_indices,
    torch::Tensor weight_indices,
    torch::Tensor lookup_table,
    torch::Tensor bias
) {
    auto device = input_indices.device();
    auto batch_size = input_indices.size(0);
    auto num_blocks = input_indices.size(1);
    auto lattice_dim = input_indices.size(2);
    auto out_features = weight_indices.size(0);
    auto lookup_table_size = lookup_table.size(0);
    
    auto output = torch::zeros({batch_size, out_features}, 
                              torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    // Flatten input and weight indices
    auto input_flat = input_indices.view({batch_size, -1});
    auto weight_flat = weight_indices.view({out_features, -1});
    
    // Choose kernel based on lookup table size
    if (lookup_table_size <= 16) {
        // Use optimized kernel for small lookup tables
        dim3 grid(out_features, batch_size);
        dim3 block(1);
        
        quantized_matmul_small_kernel<<<grid, block>>>(
            input_flat.data_ptr<int>(),
            weight_flat.data_ptr<int>(),
            lookup_table.data_ptr<float>(),
            bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size,
            out_features,
            num_blocks,
            lattice_dim,
            lookup_table_size
        );
    } else {
        // Use shared memory kernel for larger lookup tables
        dim3 grid(out_features, batch_size);
        dim3 block(256);
        size_t shared_mem_size = lookup_table_size * lookup_table_size * sizeof(float);
        
        quantized_matmul_kernel<<<grid, block, shared_mem_size>>>(
            input_flat.data_ptr<int>(),
            weight_flat.data_ptr<int>(),
            lookup_table.data_ptr<float>(),
            bias.numel() > 0 ? bias.data_ptr<float>() : nullptr,
            output.data_ptr<float>(),
            batch_size,
            out_features,
            num_blocks,
            lattice_dim,
            lookup_table_size
        );
    }
    
    cudaDeviceSynchronize();
    return output;
}