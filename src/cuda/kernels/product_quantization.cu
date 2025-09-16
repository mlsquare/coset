#include "../include/quantization.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>
#include <math.h>

// CUDA kernel implementations for product quantization operations

// Custom rounding function for lattice quantization
__device__ float custom_round_device(float x) {
    const float tiny = 1e-8f;
    float y = x - copysignf(tiny, x);
    return floorf(y + 0.5f);
}

// Product quantization kernel - processes input in blocks
__global__ void product_quantize_kernel(
    const float* input,
    float* quantized,
    int* indices,
    const float* generator_matrix,
    const float* inverse_generator,
    const float* tie_dither,
    float beta,
    float alpha,
    int batch_size,
    int input_dim,
    int lattice_dim,
    int num_blocks,
    int depth
) {
    int batch_idx = blockIdx.x;
    int block_idx = blockIdx.y;
    int thread_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || block_idx >= num_blocks) return;
    
    int input_offset = batch_idx * input_dim + block_idx * lattice_dim;
    int indices_offset = batch_idx * num_blocks * lattice_dim + block_idx * lattice_dim;
    
    // Load input block into shared memory
    extern __shared__ float shared_input[];
    if (thread_idx < lattice_dim) {
        shared_input[thread_idx] = input[input_offset + thread_idx];
    }
    __syncthreads();
    
    // Apply scaling and tie dither
    float scaled_input[8]; // Max lattice dimension
    if (thread_idx < lattice_dim) {
        float value = shared_input[thread_idx] / beta;
        if (tie_dither != nullptr) {
            value += tie_dither[thread_idx];
        }
        scaled_input[thread_idx] = value;
    }
    __syncthreads();
    
    // Compute encoding using inverse generator matrix
    if (thread_idx < lattice_dim) {
        float encoded_value = 0.0f;
        for (int j = 0; j < lattice_dim; j++) {
            encoded_value += scaled_input[j] * inverse_generator[j * lattice_dim + thread_idx];
        }
        
        // Apply modulo operation and custom rounding
        float q = (float)(1 << depth); // radix^depth
        float mod_result = fmodf(encoded_value, q);
        int quantized_value = (int)custom_round_device(mod_result);
        
        // Store quantized result
        indices[indices_offset + thread_idx] = quantized_value;
    }
    __syncthreads();
    
    // Decode back to continuous values
    if (thread_idx < lattice_dim) {
        float decoded_value = 0.0f;
        for (int j = 0; j < lattice_dim; j++) {
            decoded_value += (float)indices[indices_offset + j] * generator_matrix[j * lattice_dim + thread_idx];
        }
        
        // Scale back
        quantized[input_offset + thread_idx] = decoded_value * beta;
    }
}

// Product dequantization kernel
__global__ void product_dequantize_kernel(
    const int* indices,
    float* output,
    const float* generator_matrix,
    float beta,
    float alpha,
    int batch_size,
    int input_dim,
    int lattice_dim,
    int num_blocks,
    int depth
) {
    int batch_idx = blockIdx.x;
    int block_idx = blockIdx.y;
    int thread_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || block_idx >= num_blocks) return;
    
    int indices_offset = batch_idx * num_blocks * lattice_dim + block_idx * lattice_dim;
    int output_offset = batch_idx * input_dim + block_idx * lattice_dim;
    
    // Decode from indices using generator matrix
    if (thread_idx < lattice_dim) {
        float decoded_value = 0.0f;
        for (int j = 0; j < lattice_dim; j++) {
            decoded_value += (float)indices[indices_offset + j] * generator_matrix[j * lattice_dim + thread_idx];
        }
        
        // Scale back
        output[output_offset + thread_idx] = decoded_value * beta;
    }
}

// Product quantized matrix multiplication kernel
__global__ void product_quantized_matmul_kernel(
    const int* weight_indices,      // [output_dim, num_blocks, lattice_dim]
    const int* input_indices,       // [batch_size, num_blocks, lattice_dim]
    float* output,                  // [batch_size, output_dim]
    const float* lookup_table,
    int batch_size,
    int input_dim,
    int output_dim,
    int lattice_dim,
    int num_blocks
) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y;
    int thread_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= output_dim) return;
    
    // Shared memory for lookup table
    extern __shared__ float shared_lut[];
    
    // Load lookup table into shared memory (assuming 16x16 for 2D lattice)
    int lut_size = 16 * 16; // Max for 2D lattice with radix 4
    if (thread_idx < lut_size) {
        shared_lut[thread_idx] = lookup_table[thread_idx];
    }
    __syncthreads();
    
    // Compute dot product across all blocks
    float sum = 0.0f;
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // Get weight and input indices for this block
        int weight_offset = out_idx * num_blocks * lattice_dim + block_idx * lattice_dim;
        int input_offset = batch_idx * num_blocks * lattice_dim + block_idx * lattice_dim;
        
        // Compute dot product for this block using lookup table
        float block_sum = 0.0f;
        for (int i = 0; i < lattice_dim; i++) {
            int weight_idx = weight_indices[weight_offset + i];
            int input_idx = input_indices[input_offset + i];
            
            // Clamp indices to valid lookup table range
            weight_idx = max(0, min(weight_idx, 15));
            input_idx = max(0, min(input_idx, 15));
            
            // Use lookup table
            block_sum += shared_lut[weight_idx * 16 + input_idx];
        }
        sum += block_sum;
    }
    
    // Store result
    if (thread_idx == 0) {
        output[batch_idx * output_dim + out_idx] = sum;
    }
}

// Product quantized linear layer kernel (with bias)
__global__ void product_quantized_linear_kernel(
    const int* weight_indices,      // [output_dim, num_blocks, lattice_dim]
    const int* input_indices,       // [batch_size, num_blocks, lattice_dim]
    const float* bias,              // [output_dim]
    float* output,                  // [batch_size, output_dim]
    const float* lookup_table,
    int batch_size,
    int input_dim,
    int output_dim,
    int lattice_dim,
    int num_blocks
) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y;
    int thread_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= output_dim) return;
    
    // Shared memory for lookup table
    extern __shared__ float shared_lut[];
    
    // Load lookup table into shared memory
    int lut_size = 16 * 16; // Max for 2D lattice with radix 4
    if (thread_idx < lut_size) {
        shared_lut[thread_idx] = lookup_table[thread_idx];
    }
    __syncthreads();
    
    // Compute dot product across all blocks
    float sum = 0.0f;
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // Get weight and input indices for this block
        int weight_offset = out_idx * num_blocks * lattice_dim + block_idx * lattice_dim;
        int input_offset = batch_idx * num_blocks * lattice_dim + block_idx * lattice_dim;
        
        // Compute dot product for this block using lookup table
        float block_sum = 0.0f;
        for (int i = 0; i < lattice_dim; i++) {
            int weight_idx = weight_indices[weight_offset + i];
            int input_idx = input_indices[input_offset + i];
            
            // Clamp indices to valid lookup table range
            weight_idx = max(0, min(weight_idx, 15));
            input_idx = max(0, min(input_idx, 15));
            
            // Use lookup table
            block_sum += shared_lut[weight_idx * 16 + input_idx];
        }
        sum += block_sum;
    }
    
    // Add bias and store result
    if (thread_idx == 0) {
        output[batch_idx * output_dim + out_idx] = sum + bias[out_idx];
    }
}

// Fused product quantized linear + ReLU kernel
__global__ void fused_product_quantized_linear_relu_kernel(
    const int* weight_indices,
    const int* input_indices,
    const float* bias,
    float* output,
    const float* lookup_table,
    int batch_size,
    int input_dim,
    int output_dim,
    int lattice_dim,
    int num_blocks
) {
    int batch_idx = blockIdx.x;
    int out_idx = blockIdx.y;
    int thread_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || out_idx >= output_dim) return;
    
    // Shared memory for lookup table
    extern __shared__ float shared_lut[];
    
    // Load lookup table into shared memory
    int lut_size = 16 * 16; // Max for 2D lattice with radix 4
    if (thread_idx < lut_size) {
        shared_lut[thread_idx] = lookup_table[thread_idx];
    }
    __syncthreads();
    
    // Compute dot product across all blocks
    float sum = 0.0f;
    for (int block_idx = 0; block_idx < num_blocks; block_idx++) {
        // Get weight and input indices for this block
        int weight_offset = out_idx * num_blocks * lattice_dim + block_idx * lattice_dim;
        int input_offset = batch_idx * num_blocks * lattice_dim + block_idx * lattice_dim;
        
        // Compute dot product for this block using lookup table
        float block_sum = 0.0f;
        for (int i = 0; i < lattice_dim; i++) {
            int weight_idx = weight_indices[weight_offset + i];
            int input_idx = input_indices[input_offset + i];
            
            // Clamp indices to valid lookup table range
            weight_idx = max(0, min(weight_idx, 15));
            input_idx = max(0, min(input_idx, 15));
            
            // Use lookup table
            block_sum += shared_lut[weight_idx * 16 + input_idx];
        }
        sum += block_sum;
    }
    
    // Add bias, apply ReLU, and store result
    if (thread_idx == 0) {
        float result = sum + bias[out_idx];
        output[batch_idx * output_dim + out_idx] = fmaxf(0.0f, result);
    }
}

// Wrapper functions
extern "C" {
    void product_quantize_cuda(
        const float* input,
        float* quantized,
        int* indices,
        const float* generator_matrix,
        const float* inverse_generator,
        const float* tie_dither,
        float beta,
        float alpha,
        int batch_size,
        int input_dim,
        int lattice_dim,
        int depth,
        cudaStream_t stream
    ) {
        int num_blocks = (input_dim + lattice_dim - 1) / lattice_dim;
        dim3 grid(batch_size, num_blocks);
        dim3 block(lattice_dim);
        size_t shared_mem_size = lattice_dim * sizeof(float);
        
        product_quantize_kernel<<<grid, block, shared_mem_size, stream>>>(
            input, quantized, indices, generator_matrix, inverse_generator, tie_dither,
            beta, alpha, batch_size, input_dim, lattice_dim, num_blocks, depth
        );
    }
    
    void product_dequantize_cuda(
        const int* indices,
        float* output,
        const float* generator_matrix,
        float beta,
        float alpha,
        int batch_size,
        int input_dim,
        int lattice_dim,
        int num_blocks,
        int depth,
        cudaStream_t stream
    ) {
        dim3 grid(batch_size, num_blocks);
        dim3 block(lattice_dim);
        
        product_dequantize_kernel<<<grid, block, 0, stream>>>(
            indices, output, generator_matrix, beta, alpha,
            batch_size, input_dim, lattice_dim, num_blocks, depth
        );
    }
    
    void product_quantized_matmul_cuda(
        const int* weight_indices,
        const int* input_indices,
        float* output,
        const float* lookup_table,
        int batch_size,
        int input_dim,
        int output_dim,
        int lattice_dim,
        int num_blocks,
        cudaStream_t stream
    ) {
        dim3 grid(batch_size, output_dim);
        dim3 block(256);
        size_t shared_mem_size = 16 * 16 * sizeof(float); // 2D lattice lookup table
        
        product_quantized_matmul_kernel<<<grid, block, shared_mem_size, stream>>>(
            weight_indices, input_indices, output, lookup_table,
            batch_size, input_dim, output_dim, lattice_dim, num_blocks
        );
    }
    
    void product_quantized_linear_cuda(
        const int* weight_indices,
        const int* input_indices,
        const float* bias,
        float* output,
        const float* lookup_table,
        int batch_size,
        int input_dim,
        int output_dim,
        int lattice_dim,
        int num_blocks,
        cudaStream_t stream
    ) {
        dim3 grid(batch_size, output_dim);
        dim3 block(256);
        size_t shared_mem_size = 16 * 16 * sizeof(float); // 2D lattice lookup table
        
        product_quantized_linear_kernel<<<grid, block, shared_mem_size, stream>>>(
            weight_indices, input_indices, bias, output, lookup_table,
            batch_size, input_dim, output_dim, lattice_dim, num_blocks
        );
    }
    
    void fused_product_quantized_linear_relu_cuda(
        const int* weight_indices,
        const int* input_indices,
        const float* bias,
        float* output,
        const float* lookup_table,
        int batch_size,
        int input_dim,
        int output_dim,
        int lattice_dim,
        int num_blocks,
        cudaStream_t stream
    ) {
        dim3 grid(batch_size, output_dim);
        dim3 block(256);
        size_t shared_mem_size = 16 * 16 * sizeof(float); // 2D lattice lookup table
        
        fused_product_quantized_linear_relu_kernel<<<grid, block, shared_mem_size, stream>>>(
            weight_indices, input_indices, bias, output, lookup_table,
            batch_size, input_dim, output_dim, lattice_dim, num_blocks
        );
    }
}
