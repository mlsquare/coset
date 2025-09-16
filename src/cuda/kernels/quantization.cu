#include "../include/quantization.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>

// CUDA kernel implementations for quantization operations

__global__ void lattice_quantize_kernel(
    const float* input,
    int* indices,
    const float* codebook,
    const float* scales,
    const int* zero_points,
    int batch_size,
    int seq_len,
    int lattice_dim,
    int num_codewords,
    int depth
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int thread_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    int input_offset = (batch_idx * seq_len + seq_idx) * lattice_dim;
    int output_offset = batch_idx * seq_len + seq_idx;
    
    // Get quantization parameters for this depth
    float scale = scales[depth];
    int zero_point = zero_points[depth];
    
    // Load input vector into shared memory
    extern __shared__ float shared_input[];
    if (thread_idx < lattice_dim) {
        shared_input[thread_idx] = input[input_offset + thread_idx];
    }
    __syncthreads();
    
    // Find closest lattice point
    float min_distance = FLT_MAX;
    int best_index = 0;
    
    for (int i = 0; i < num_codewords; i++) {
        float distance = 0.0f;
        
        // Compute distance to lattice point
        for (int j = 0; j < lattice_dim; j++) {
            float diff = (shared_input[j] / scale + zero_point) - codebook[i * lattice_dim + j];
            distance += diff * diff;
        }
        
        if (distance < min_distance) {
            min_distance = distance;
            best_index = i;
        }
    }
    
    // Store result
    if (thread_idx == 0) {
        indices[output_offset] = best_index;
    }
}

__global__ void lattice_dequantize_kernel(
    const int* indices,
    float* output,
    const float* codebook,
    const float* scales,
    const int* zero_points,
    int batch_size,
    int seq_len,
    int lattice_dim,
    int depth
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int thread_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    int input_offset = batch_idx * seq_len + seq_idx;
    int output_offset = (batch_idx * seq_len + seq_idx) * lattice_dim;
    
    // Get quantization parameters
    float scale = scales[depth];
    int zero_point = zero_points[depth];
    int index = indices[input_offset];
    
    // Dequantize
    if (thread_idx < lattice_dim) {
        float quantized_value = codebook[index * lattice_dim + thread_idx];
        output[output_offset + thread_idx] = (quantized_value - zero_point) * scale;
    }
}

__global__ void hierarchical_quantize_kernel(
    const float* input,
    int* indices,
    const float* codebook,
    const float* scales,
    const int* zero_points,
    const float* hierarchy_weights,
    int batch_size,
    int seq_len,
    int lattice_dim,
    int num_levels
) {
    int batch_idx = blockIdx.x;
    int seq_idx = blockIdx.y;
    int level_idx = blockIdx.z;
    int thread_idx = threadIdx.x;
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || level_idx >= num_levels) return;
    
    int input_offset = (batch_idx * seq_len + seq_idx) * lattice_dim;
    int output_offset = (batch_idx * seq_len + seq_idx) * num_levels + level_idx;
    
    // Get quantization parameters for this level
    float scale = scales[level_idx];
    int zero_point = zero_points[level_idx];
    
    // Load input vector into shared memory
    extern __shared__ float shared_input[];
    if (thread_idx < lattice_dim) {
        shared_input[thread_idx] = input[input_offset + thread_idx];
    }
    __syncthreads();
    
    // Find closest lattice point
    float min_distance = FLT_MAX;
    int best_index = 0;
    
    for (int i = 0; i < (1 << lattice_dim); i++) {
        float distance = 0.0f;
        
        // Compute distance to lattice point
        for (int j = 0; j < lattice_dim; j++) {
            float diff = (shared_input[j] / scale + zero_point) - codebook[i * lattice_dim + j];
            distance += diff * diff;
        }
        
        if (distance < min_distance) {
            min_distance = distance;
            best_index = i;
        }
    }
    
    // Store result
    if (thread_idx == 0) {
        indices[output_offset] = best_index;
    }
}

__global__ void radixq_encode_kernel(
    const float* input,
    int* encoded,
    int radix,
    int depth,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    
    // Convert to radix-q representation
    int value = (int)input[idx];
    int encoded_value = 0;
    
    for (int i = 0; i < depth; i++) {
        encoded_value += (value % radix) * (int)powf(radix, i);
        value = value / radix;
    }
    
    encoded[idx] = encoded_value;
}

__global__ void radixq_decode_kernel(
    const int* encoded,
    float* output,
    int radix,
    int depth,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    
    // Convert from radix-q representation
    int encoded_value = encoded[idx];
    int decoded_value = 0;
    
    for (int i = 0; i < depth; i++) {
        decoded_value += (encoded_value % radix) * (int)powf(radix, i);
        encoded_value = encoded_value / radix;
    }
    
    output[idx] = decoded_value;
}

__global__ void gradient_quantize_kernel(
    const float* gradients,
    int* quantized_gradients,
    const float* codebook,
    const float* scales,
    int size,
    int depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    
    // Get quantization parameters
    float scale = scales[depth];
    
    // Quantize gradient
    float normalized = gradients[idx] / scale;
    int quantized = (int)roundf(normalized);
    
    // Clamp to valid range
    int max_value = (1 << 8) - 1;  // 8-bit quantization
    quantized = max(0, min(quantized, max_value));
    
    quantized_gradients[idx] = quantized;
}

__global__ void quantized_gradient_accumulate_kernel(
    int* accumulated_gradients,
    const int* new_gradients,
    int size,
    int depth
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= size) return;
    
    // Accumulate in quantized space
    accumulated_gradients[idx] += new_gradients[idx];
}

// Wrapper functions
extern "C" {
    void lattice_quantize_cuda(
        const float* input,
        int* indices,
        const float* codebook,
        const float* scales,
        const int* zero_points,
        int batch_size,
        int seq_len,
        int lattice_dim,
        int num_codewords,
        int depth,
        cudaStream_t stream
    ) {
        dim3 grid(batch_size, seq_len);
        dim3 block(lattice_dim);
        size_t shared_mem_size = lattice_dim * sizeof(float);
        
        lattice_quantize_kernel<<<grid, block, shared_mem_size, stream>>>(
            input, indices, codebook, scales, zero_points,
            batch_size, seq_len, lattice_dim, num_codewords, depth
        );
    }
    
    void lattice_dequantize_cuda(
        const int* indices,
        float* output,
        const float* codebook,
        const float* scales,
        const int* zero_points,
        int batch_size,
        int seq_len,
        int lattice_dim,
        int depth,
        cudaStream_t stream
    ) {
        dim3 grid(batch_size, seq_len);
        dim3 block(lattice_dim);
        
        lattice_dequantize_kernel<<<grid, block, 0, stream>>>(
            indices, output, codebook, scales, zero_points,
            batch_size, seq_len, lattice_dim, depth
        );
    }
    
    void hierarchical_quantize_cuda(
        const float* input,
        int* indices,
        const float* codebook,
        const float* scales,
        const int* zero_points,
        const float* hierarchy_weights,
        int batch_size,
        int seq_len,
        int lattice_dim,
        int num_levels,
        cudaStream_t stream
    ) {
        dim3 grid(batch_size, seq_len, num_levels);
        dim3 block(lattice_dim);
        size_t shared_mem_size = lattice_dim * sizeof(float);
        
        hierarchical_quantize_kernel<<<grid, block, shared_mem_size, stream>>>(
            input, indices, codebook, scales, zero_points, hierarchy_weights,
            batch_size, seq_len, lattice_dim, num_levels
        );
    }
    
    void radixq_encode_cuda(
        const float* input,
        int* encoded,
        int radix,
        int depth,
        int size,
        cudaStream_t stream
    ) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        
        radixq_encode_kernel<<<grid_size, block_size, 0, stream>>>(
            input, encoded, radix, depth, size
        );
    }
    
    void radixq_decode_cuda(
        const int* encoded,
        float* output,
        int radix,
        int depth,
        int size,
        cudaStream_t stream
    ) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        
        radixq_decode_kernel<<<grid_size, block_size, 0, stream>>>(
            encoded, output, radix, depth, size
        );
    }
    
    void gradient_quantize_cuda(
        const float* gradients,
        int* quantized_gradients,
        const float* codebook,
        const float* scales,
        int size,
        int depth,
        cudaStream_t stream
    ) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        
        gradient_quantize_kernel<<<grid_size, block_size, 0, stream>>>(
            gradients, quantized_gradients, codebook, scales, size, depth
        );
    }
    
    void quantized_gradient_accumulate_cuda(
        int* accumulated_gradients,
        const int* new_gradients,
        int size,
        int depth,
        cudaStream_t stream
    ) {
        int block_size = 256;
        int grid_size = (size + block_size - 1) / block_size;
        
        quantized_gradient_accumulate_kernel<<<grid_size, block_size, 0, stream>>>(
            accumulated_gradients, new_gradients, size, depth
        );
    }
}
