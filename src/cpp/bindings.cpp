#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "operators.h"
#include "autograd.h"

// PyTorch C++ extension bindings for CoSet

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "CoSet: Hierarchical Nested Lattice Quantization for Matrix Operations";
    
    // Product quantization operations
    m.def("product_quantize", &product_quantize_cuda, "Product quantization for arbitrary input dimensions");
    m.def("product_dequantize", &product_dequantize_cuda, "Product dequantization from block indices");
    
    // Core quantization operations (legacy)
    m.def("lattice_quantize", &lattice_quantize_cuda, "Quantize tensor to lattice points");
    m.def("lattice_dequantize", &lattice_dequantize_cuda, "Dequantize lattice indices to continuous values");
    m.def("hierarchical_quantize", &hierarchical_quantize_cuda, "Hierarchical quantization");
    m.def("hierarchical_dequantize", &hierarchical_dequantize_cuda, "Hierarchical dequantization");
    
    // Radix-q operations
    m.def("radixq_encode", &radixq_encode_cuda, "Encode using radix-q representation");
    m.def("radixq_decode", &radixq_decode_cuda, "Decode from radix-q representation");
    
    // Product quantization matrix operations
    m.def("product_quantized_matmul", &product_quantized_matmul_cuda, "Product quantized matrix multiplication");
    m.def("product_quantized_linear", &product_quantized_linear_cuda, "Product quantized linear layer");
    m.def("fused_product_quantized_linear_relu", &fused_product_quantized_linear_relu_cuda, "Fused product quantized linear + ReLU");
    
    // Matrix operations (legacy)
    m.def("lookup_dot_product", &lookup_dot_product_cuda, "Dot product using lookup tables");
    m.def("quantized_matmul", &quantized_matmul_cuda, "Quantized matrix multiplication");
    m.def("batched_quantized_matmul", &batched_quantized_matmul_cuda, "Batched quantized matrix multiplication");
    m.def("fused_quantized_linear_relu", &fused_quantized_linear_relu_cuda, "Fused quantized linear + ReLU");
    m.def("tiled_quantized_matmul", &tiled_quantized_matmul_cuda, "Tiled quantized matrix multiplication");
    
    // Gradient operations
    m.def("gradient_quantize", &gradient_quantize_cuda, "Quantize gradients for communication");
    m.def("quantized_gradient_accumulate", &quantized_gradient_accumulate_cuda, "Accumulate quantized gradients");
    m.def("quantized_allreduce", &quantized_allreduce_cuda, "All-reduce in quantized space");
    
    // Autograd functions
    py::class_<QuantizedLinearFunction>(m, "QuantizedLinearFunction")
        .def_static("forward", &QuantizedLinearFunction::forward)
        .def_static("backward", &QuantizedLinearFunction::backward);
    
    py::class_<QuantizedMatMulFunction>(m, "QuantizedMatMulFunction")
        .def_static("forward", &QuantizedMatMulFunction::forward)
        .def_static("backward", &QuantizedMatMulFunction::backward);
    
    py::class_<STEFunction>(m, "STEFunction")
        .def_static("forward", &STEFunction::forward)
        .def_static("backward", &STEFunction::backward);
    
    // Convenience functions
    m.def("ste_quantize", &ste_quantize, "Apply straight-through estimator to quantization");
    m.def("quantized_linear", &quantized_linear, "Apply quantized linear transformation");
    m.def("quantized_matmul", &quantized_matmul, "Apply quantized matrix multiplication");
    m.def("quantize_gradients", &quantize_gradients, "Quantize gradients for communication");
}
