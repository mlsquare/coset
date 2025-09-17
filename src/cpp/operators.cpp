#include <torch/extension.h>
#include <vector>

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

// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quantized_matmul_cuda", &quantized_matmul_cuda, "Quantized matrix multiplication CUDA kernel");
    m.def("vectorized_quantize_cuda", &vectorized_quantize_cuda, "Vectorized quantization CUDA kernel");
    m.def("closest_point_e8_cuda", &closest_point_e8_cuda, "E8 closest point CUDA kernel");
    m.def("vectorized_product_quantize_cuda", &vectorized_product_quantize_cuda, "Vectorized product quantization CUDA kernel");
    m.def("batch_quantize_cuda", &batch_quantize_cuda, "Batch quantization CUDA kernel");
    
    // Utility functions
    m.def("elementwise_add_cuda", &elementwise_add_cuda, "Elementwise addition CUDA kernel");
    m.def("elementwise_multiply_cuda", &elementwise_multiply_cuda, "Elementwise multiplication CUDA kernel");
    m.def("elementwise_divide_cuda", &elementwise_divide_cuda, "Elementwise division CUDA kernel");
}
