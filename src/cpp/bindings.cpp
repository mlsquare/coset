#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "operators.h"

// PyTorch C++ extension bindings for CoSet

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CoSet: Hierarchical Nested Lattice Quantization for Matrix Operations";
    
    // Core quantization operations
    m.def("quantized_matmul_cuda", &quantized_matmul_cuda, "Quantized matrix multiplication CUDA kernel");
    m.def("vectorized_quantize_cuda", &vectorized_quantize_cuda, "Vectorized quantization CUDA kernel");
    m.def("closest_point_e8_cuda", &closest_point_e8_cuda, "E8 closest point CUDA kernel");
    m.def("vectorized_product_quantize_cuda", &vectorized_product_quantize_cuda, "Vectorized product quantization CUDA kernel");
    m.def("batch_quantize_cuda", &batch_quantize_cuda, "Batch quantization CUDA kernel");
    
    // Utility functions
    m.def("elementwise_add_cuda", &elementwise_add_cuda, "Elementwise addition CUDA kernel");
    m.def("elementwise_multiply_cuda", &elementwise_multiply_cuda, "Elementwise multiplication CUDA kernel");
    m.def("elementwise_divide_cuda", &elementwise_divide_cuda, "Elementwise division CUDA kernel");
    
    // Gradient functions
    m.def("quantized_matmul_backward_cuda", &quantized_matmul_backward_cuda, "Quantized matmul backward CUDA kernel");
    m.def("quantization_backward_cuda", &quantization_backward_cuda, "Quantization backward CUDA kernel");
}
