#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Vectorized product quantization kernel
__global__ void vectorized_product_quantize_kernel(
    const float* input,
    float* output,
    const float* generator_matrix,
    const float* inverse_generator_matrix,
    const float* eps,
    int* indices,
    int batch_size,
    int input_dim,
    int lattice_dim,
    int num_blocks,
    float beta,
    int q
) {
    int block_idx = blockIdx.x;
    int batch_idx = blockIdx.y;
    int thread_id = threadIdx.x;
    
    if (block_idx >= num_blocks || batch_idx >= batch_size) return;
    
    // Process lattice_dim elements per block
    float block_input[8];  // E8 lattice dimension
    float block_output[8];
    int block_indices[8];
    
    // Load input block
    for (int i = thread_id; i < lattice_dim; i += blockDim.x) {
        int global_idx = batch_idx * input_dim + block_idx * lattice_dim + i;
        if (global_idx < batch_idx * input_dim + input_dim) {
            block_input[i] = input[global_idx];
        } else {
            block_input[i] = 0.0f;  // Padding
        }
    }
    __syncthreads();
    
    // Scale by beta
    for (int i = thread_id; i < lattice_dim; i += blockDim.x) {
        block_input[i] /= beta;
    }
    __syncthreads();
    
    // Add epsilon
    for (int i = thread_id; i < lattice_dim; i += blockDim.x) {
        block_input[i] += eps[i];
    }
    __syncthreads();
    
    // Find closest point (simplified E8 implementation)
    for (int i = thread_id; i < lattice_dim; i += blockDim.x) {
        block_output[i] = floorf(block_input[i] + 0.5f);
    }
    __syncthreads();
    
    // Matrix multiplication with inverse generator matrix
    for (int i = thread_id; i < lattice_dim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < lattice_dim; j++) {
            sum += block_output[j] * inverse_generator_matrix[j * lattice_dim + i];
        }
        block_indices[i] = (int)fmodf(sum, (float)q);
    }
    __syncthreads();
    
    // Matrix multiplication with generator matrix for output
    for (int i = thread_id; i < lattice_dim; i += blockDim.x) {
        float sum = 0.0f;
        for (int j = 0; j < lattice_dim; j++) {
            sum += (float)block_indices[j] * generator_matrix[j * lattice_dim + i];
        }
        block_output[i] = sum * beta;
    }
    __syncthreads();
    
    // Store results
    for (int i = thread_id; i < lattice_dim; i += blockDim.x) {
        int global_idx = batch_idx * input_dim + block_idx * lattice_dim + i;
        if (global_idx < batch_idx * input_dim + input_dim) {
            output[global_idx] = block_output[i];
            indices[batch_idx * num_blocks * lattice_dim + block_idx * lattice_dim + i] = block_indices[i];
        }
    }
}

// Batch quantization kernel for multiple inputs
__global__ void batch_quantize_kernel(
    const float* inputs,
    float* outputs,
    int* indices,
    const float* generator_matrix,
    const float* inverse_generator_matrix,
    const float* eps,
    int batch_size,
    int input_dim,
    int lattice_dim,
    float beta,
    int q
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    const float* input = inputs + idx * input_dim;
    float* output = outputs + idx * input_dim;
    int* input_indices = indices + idx * input_dim;
    
    int num_blocks = (input_dim + lattice_dim - 1) / lattice_dim;
    
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        float block_input[8];
        float block_output[8];
        int block_indices[8];
        
        // Load and pad block
        for (int i = 0; i < lattice_dim; i++) {
            int global_idx = block_idx * lattice_dim + i;
            if (global_idx < input_dim) {
                block_input[i] = input[global_idx] / beta + eps[i];
            } else {
                block_input[i] = eps[i];  // Padding
            }
        }
        
        // Find closest point
        for (int i = 0; i < lattice_dim; i++) {
            block_output[i] = floorf(block_input[i] + 0.5f);
        }
        
        // Compute indices
        for (int i = 0; i < lattice_dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < lattice_dim; j++) {
                sum += block_output[j] * inverse_generator_matrix[j * lattice_dim + i];
            }
            block_indices[i] = (int)fmodf(sum, (float)q);
        }
        
        // Compute output
        for (int i = 0; i < lattice_dim; i++) {
            float sum = 0.0f;
            for (int j = 0; j < lattice_dim; j++) {
                sum += (float)block_indices[j] * generator_matrix[j * lattice_dim + i];
            }
            block_output[i] = sum * beta;
        }
        
        // Store results
        for (int i = 0; i < lattice_dim; i++) {
            int global_idx = block_idx * lattice_dim + i;
            if (global_idx < input_dim) {
                output[global_idx] = block_output[i];
                input_indices[global_idx] = block_indices[i];
            }
        }
    }
}

torch::Tensor vectorized_product_quantize_cuda(
    torch::Tensor input,
    torch::Tensor generator_matrix,
    torch::Tensor inverse_generator_matrix,
    torch::Tensor eps,
    float beta,
    int q
) {
    auto device = input.device();
    auto batch_size = input.size(0);
    auto input_dim = input.size(1);
    auto lattice_dim = generator_matrix.size(0);
    
    auto output = torch::zeros_like(input);
    auto indices = torch::zeros({batch_size, (input_dim + lattice_dim - 1) / lattice_dim, lattice_dim}, 
                               torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    int num_blocks = (input_dim + lattice_dim - 1) / lattice_dim;
    
    // Launch kernel
    dim3 grid(num_blocks, batch_size);
    dim3 block(256);
    
    vectorized_product_quantize_kernel<<<grid, block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        generator_matrix.data_ptr<float>(),
        inverse_generator_matrix.data_ptr<float>(),
        eps.data_ptr<float>(),
        indices.data_ptr<int>(),
        batch_size,
        input_dim,
        lattice_dim,
        num_blocks,
        beta,
        q
    );
    
    cudaDeviceSynchronize();
    return output;
}

torch::Tensor batch_quantize_cuda(
    torch::Tensor inputs,
    torch::Tensor generator_matrix,
    torch::Tensor inverse_generator_matrix,
    torch::Tensor eps,
    float beta,
    int q
) {
    auto device = inputs.device();
    auto batch_size = inputs.size(0);
    auto input_dim = inputs.size(1);
    auto lattice_dim = generator_matrix.size(0);
    
    auto outputs = torch::zeros_like(inputs);
    auto indices = torch::zeros_like(inputs, torch::TensorOptions().dtype(torch::kInt32));
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    batch_quantize_kernel<<<blocks, threads_per_block>>>(
        inputs.data_ptr<float>(),
        outputs.data_ptr<float>(),
        indices.data_ptr<int>(),
        generator_matrix.data_ptr<float>(),
        inverse_generator_matrix.data_ptr<float>(),
        eps.data_ptr<float>(),
        batch_size,
        input_dim,
        lattice_dim,
        beta,
        q
    );
    
    cudaDeviceSynchronize();
    return outputs;
}