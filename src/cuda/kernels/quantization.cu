#include <torch/extension.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>

// CUDA kernel for E8 closest point function
__device__ float custom_round_cuda(float x) {
    return floorf(x + 0.5f);
}

__device__ float norm_squared_cuda(const float* x, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; i++) {
        sum += x[i] * x[i];
    }
    return sum;
}

__device__ void g_x_cuda(const float* x, float* g_x, int dim) {
    float f_x[8];  // E8 lattice has dimension 8
    float delta[8];
    
    // Compute f_x and delta
    for (int i = 0; i < dim; i++) {
        f_x[i] = custom_round_cuda(x[i]);
        delta[i] = fabsf(x[i] - f_x[i]);
    }
    
    // Find maximum delta
    int k = 0;
    float max_delta = delta[0];
    for (int i = 1; i < dim; i++) {
        if (delta[i] > max_delta) {
            max_delta = delta[i];
            k = i;
        }
    }
    
    // Copy f_x to g_x
    for (int i = 0; i < dim; i++) {
        g_x[i] = f_x[i];
    }
    
    // Update g_x[k]
    float x_k = x[k];
    float f_x_k = f_x[k];
    
    if (x_k >= 0.0f) {
        g_x[k] = (f_x_k < x_k) ? (f_x_k + 1.0f) : (f_x_k - 1.0f);
    } else {
        g_x[k] = (f_x_k <= x_k) ? (f_x_k + 1.0f) : (f_x_k - 1.0f);
    }
}

__global__ void closest_point_e8_kernel(
    const float* input,
    float* output,
    int batch_size,
    int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    const float* x = input + idx * dim;
    float* y = output + idx * dim;
    
    // Compute f_x
    float f_x[8];
    float sum_f_x = 0.0f;
    for (int i = 0; i < dim; i++) {
        f_x[i] = custom_round_cuda(x[i]);
        sum_f_x += f_x[i];
    }
    
    // Compute y_0
    float y_0[8];
    if (fmodf(sum_f_x, 2.0f) == 0.0f) {
        for (int i = 0; i < dim; i++) {
            y_0[i] = f_x[i];
        }
    } else {
        g_x_cuda(x, y_0, dim);
    }
    
    // Compute f_x_shifted and g_x_shifted
    float x_shifted[8];
    for (int i = 0; i < dim; i++) {
        x_shifted[i] = x[i] - 0.5f;
    }
    
    float f_x_shifted[8];
    float sum_f_x_shifted = 0.0f;
    for (int i = 0; i < dim; i++) {
        f_x_shifted[i] = custom_round_cuda(x_shifted[i]);
        sum_f_x_shifted += f_x_shifted[i];
    }
    
    float g_x_shifted[8];
    g_x_cuda(x_shifted, g_x_shifted, dim);
    
    // Compute y_1
    float y_1[8];
    if (fmodf(sum_f_x_shifted, 2.0f) == 0.0f) {
        for (int i = 0; i < dim; i++) {
            y_1[i] = f_x_shifted[i] + 0.5f;
        }
    } else {
        for (int i = 0; i < dim; i++) {
            y_1[i] = g_x_shifted[i] + 0.5f;
        }
    }
    
    // Choose closest point
    float norm_y_0 = norm_squared_cuda(y_0, dim);
    float norm_y_1 = norm_squared_cuda(y_1, dim);
    
    if (norm_y_0 < norm_y_1) {
        for (int i = 0; i < dim; i++) {
            y[i] = y_0[i];
        }
    } else {
        for (int i = 0; i < dim; i++) {
            y[i] = y_1[i];
        }
    }
}

// Vectorized quantization kernel
__global__ void vectorized_quantize_kernel(
    const float* input,
    const float* generator_matrix,
    const float* inverse_generator_matrix,
    const float* eps,
    int* output_indices,
    int batch_size,
    int lattice_dim,
    float beta,
    int q
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= batch_size) return;
    
    const float* x = input + idx * lattice_dim;
    int* indices = output_indices + idx * lattice_dim;
    
    // Scale by beta
    float x_scaled[8];
    for (int i = 0; i < lattice_dim; i++) {
        x_scaled[i] = x[i] / beta;
    }
    
    // Add epsilon
    float x_with_eps[8];
    for (int i = 0; i < lattice_dim; i++) {
        x_with_eps[i] = x_scaled[i] + eps[i];
    }
    
    // Find closest point (simplified for now)
    float closest_point[8];
    for (int i = 0; i < lattice_dim; i++) {
        closest_point[i] = custom_round_cuda(x_with_eps[i]);
    }
    
    // Matrix multiplication with inverse generator matrix
    for (int i = 0; i < lattice_dim; i++) {
        float sum = 0.0f;
        for (int j = 0; j < lattice_dim; j++) {
            sum += closest_point[j] * inverse_generator_matrix[j * lattice_dim + i];
        }
        indices[i] = (int)fmodf(sum, (float)q);
    }
}

torch::Tensor closest_point_e8_cuda(torch::Tensor input) {
    auto device = input.device();
    auto batch_size = input.size(0);
    auto dim = input.size(1);
    
    auto output = torch::zeros_like(input);
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    closest_point_e8_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        dim
    );
    
    cudaDeviceSynchronize();
    return output;
}

torch::Tensor vectorized_quantize_cuda(
    torch::Tensor input,
    torch::Tensor generator_matrix,
    torch::Tensor inverse_generator_matrix,
    torch::Tensor eps,
    float beta,
    int q
) {
    auto device = input.device();
    auto batch_size = input.size(0);
    auto lattice_dim = input.size(1);
    
    auto output_indices = torch::zeros({batch_size, lattice_dim}, 
                                      torch::TensorOptions().dtype(torch::kInt32).device(device));
    
    // Launch kernel
    int threads_per_block = 256;
    int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    vectorized_quantize_kernel<<<blocks, threads_per_block>>>(
        input.data_ptr<float>(),
        generator_matrix.data_ptr<float>(),
        inverse_generator_matrix.data_ptr<float>(),
        eps.data_ptr<float>(),
        output_indices.data_ptr<int>(),
        batch_size,
        lattice_dim,
        beta,
        q
    );
    
    cudaDeviceSynchronize();
    return output_indices;
}