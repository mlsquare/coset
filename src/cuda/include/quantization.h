#pragma once

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

// Core quantization structures
struct LatticeConfig {
    int lattice_type;
    int radix;
    int num_layers;
    int lattice_dim;
    bool learnable_scales;
    bool learnable_zero_points;
};

struct LatticeCodebook {
    float* codewords;           // [num_codewords, lattice_dim]
    int* indices;              // [num_codewords]
    float* scales;             // [num_layers]
    int* zero_points;          // [num_layers]
    float* hierarchy_weights;  // [num_layers]
    
    int lattice_dim;
    int num_codewords;
    int num_layers;
    int radix;
};

// CUDA kernel declarations
extern "C" {
    // Quantization kernels
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
    );
    
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
    );
    
    // Hierarchical quantization kernels
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
    );
    
    // Radix-q encoding/decoding kernels
    void radixq_encode_cuda(
        const float* input,
        int* encoded,
        int radix,
        int depth,
        int size,
        cudaStream_t stream
    );
    
    void radixq_decode_cuda(
        const int* encoded,
        float* output,
        int radix,
        int depth,
        int size,
        cudaStream_t stream
    );
    
    // Matrix operation kernels
    void lookup_dot_product_cuda(
        const int* x_indices,
        const int* y_indices,
        float* output,
        const float* lookup_table,
        int batch_size,
        int seq_len,
        cudaStream_t stream
    );
    
    void quantized_matmul_cuda(
        const int* weight_indices,
        const int* input_indices,
        float* output,
        const float* lookup_table,
        int batch_size,
        int seq_len,
        int hidden_dim,
        int output_dim,
        cudaStream_t stream
    );
    
    // Gradient operation kernels
    void gradient_quantize_cuda(
        const float* gradients,
        int* quantized_gradients,
        const float* codebook,
        const float* scales,
        int size,
        int depth,
        cudaStream_t stream
    );
    
    void quantized_gradient_accumulate_cuda(
        int* accumulated_gradients,
        const int* new_gradients,
        int size,
        int depth,
        cudaStream_t stream
    );
    
    void quantized_allreduce_cuda(
        int* gradients,
        int* temp_buffer,
        int size,
        int depth,
        int num_ranks,
        cudaStream_t stream
    );
}
