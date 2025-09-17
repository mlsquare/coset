#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Gradient computation kernels for backpropagation

__global__ void quantized_matmul_backward_kernel(
    const float* grad_output,
    const int* input_indices,
    const int* weight_indices,
    const float* lookup_table,
    float* grad_input,
    float* grad_weight,
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
    
    float grad_val = grad_output[batch_idx * out_features + out_idx];
    int total_elements = num_blocks * lattice_dim;
    
    // Compute gradients
    for (int elem_idx = thread_id; elem_idx < total_elements; elem_idx += blockDim.x) {
        int input_idx = input_indices[batch_idx * total_elements + elem_idx];
        int weight_idx = weight_indices[out_idx * total_elements + elem_idx];
        
        // Clamp indices
        input_idx = max(0, min(input_idx, lookup_table_size - 1));
        weight_idx = max(0, min(weight_idx, lookup_table_size - 1));
        
        // Get lookup table gradient
        float lookup_grad = lookup_table[input_idx * lookup_table_size + weight_idx];
        
        // Accumulate gradients (simplified straight-through estimator)
        atomicAdd(&grad_input[batch_idx * total_elements + elem_idx], grad_val * lookup_grad);
        atomicAdd(&grad_weight[out_idx * total_elements + elem_idx], grad_val * lookup_grad);
    }
}

__global__ void quantization_backward_kernel(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Straight-through estimator: pass gradient through
        grad_input[idx] = grad_output[idx];
    }
}

torch::Tensor quantized_matmul_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input_indices,
    torch::Tensor weight_indices,
    torch::Tensor lookup_table
) {
    auto device = grad_output.device();
    auto batch_size = input_indices.size(0);
    auto num_blocks = input_indices.size(1);
    auto lattice_dim = input_indices.size(2);
    auto out_features = weight_indices.size(0);
    auto lookup_table_size = lookup_table.size(0);
    
    auto grad_input = torch::zeros_like(input_indices, torch::TensorOptions().dtype(torch::kFloat32));
    auto grad_weight = torch::zeros_like(weight_indices, torch::TensorOptions().dtype(torch::kFloat32));
    
    // Flatten tensors
    auto input_flat = input_indices.view({batch_size, -1});
    auto weight_flat = weight_indices.view({out_features, -1});
    auto grad_input_flat = grad_input.view({batch_size, -1});
    auto grad_weight_flat = grad_weight.view({out_features, -1});
    
    // Launch kernel
    dim3 grid(out_features, batch_size);
    dim3 block(256);
    
    quantized_matmul_backward_kernel<<<grid, block>>>(
        grad_output.data_ptr<float>(),
        input_flat.data_ptr<int>(),
        weight_flat.data_ptr<int>(),
        lookup_table.data_ptr<float>(),
        grad_input_flat.data_ptr<float>(),
        grad_weight_flat.data_ptr<float>(),
        batch_size,
        out_features,
        num_blocks,
        lattice_dim,
        lookup_table_size
    );
    
    cudaDeviceSynchronize();
    
    // Return gradient tensors
    return torch::stack({grad_input, grad_weight});
}

torch::Tensor quantization_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input
) {
    auto grad_input = torch::zeros_like(input);
    int size = input.numel();
    int threads_per_block = 256;
    int blocks = (size + threads_per_block - 1) / threads_per_block;
    
    quantization_backward_kernel<<<blocks, threads_per_block>>>(
        grad_output.data_ptr<float>(),
        input.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        size
    );
    
    cudaDeviceSynchronize();
    return grad_input;
}
