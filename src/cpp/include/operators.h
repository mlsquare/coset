#pragma once
#include <torch/extension.h>

// CUDA kernel declarations
torch::Tensor quantized_matmul_cuda(
    torch::Tensor input_indices,
    torch::Tensor weight_indices,
    torch::Tensor lookup_table,
    torch::Tensor bias
);

torch::Tensor vectorized_quantize_cuda(
    torch::Tensor input,
    torch::Tensor generator_matrix,
    torch::Tensor inverse_generator_matrix,
    torch::Tensor eps,
    float beta,
    int q
);

torch::Tensor closest_point_e8_cuda(
    torch::Tensor input
);

torch::Tensor vectorized_product_quantize_cuda(
    torch::Tensor input,
    torch::Tensor generator_matrix,
    torch::Tensor inverse_generator_matrix,
    torch::Tensor eps,
    float beta,
    int q
);

torch::Tensor batch_quantize_cuda(
    torch::Tensor inputs,
    torch::Tensor generator_matrix,
    torch::Tensor inverse_generator_matrix,
    torch::Tensor eps,
    float beta,
    int q
);

// Utility functions
torch::Tensor elementwise_add_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor elementwise_multiply_cuda(torch::Tensor a, torch::Tensor b);
torch::Tensor elementwise_divide_cuda(torch::Tensor a, torch::Tensor b);

// Gradient functions
torch::Tensor quantized_matmul_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input_indices,
    torch::Tensor weight_indices,
    torch::Tensor lookup_table
);

torch::Tensor quantization_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input
);
