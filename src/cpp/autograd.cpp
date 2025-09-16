#include <torch/extension.h>
#include <torch/autograd.h>
#include <ATen/cuda/CUDAContext.h>

#include "operators.h"

// Autograd function implementations

// Straight-Through Estimator (STE) function
class STEFunction : public torch::autograd::Function<STEFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input,
        const torch::Tensor& quantized
    ) {
        ctx->save_for_backward({input, quantized});
        return quantized;
    }
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto quantized = saved[1];
        
        // Use identity function for gradients (STE)
        return {grad_outputs[0], torch::Tensor()};
    }
};

// Quantized linear function
class QuantizedLinearFunction : public torch::autograd::Function<QuantizedLinearFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input,
        const torch::Tensor& weight,
        const torch::Tensor& bias,
        const torch::Tensor& codebook,
        const torch::Tensor& scales,
        const torch::Tensor& zero_points,
        int depth
    ) {
        // Save tensors for backward pass
        ctx->save_for_backward({input, weight, bias, codebook, scales, zero_points});
        ctx->saved_data["depth"] = depth;
        
        // Quantize input
        auto input_indices = lattice_quantize_cuda(input, codebook, scales, zero_points, depth);
        
        // Quantize weights
        auto weight_indices = lattice_quantize_cuda(weight, codebook, scales, zero_points, depth);
        
        // Perform quantized matrix multiplication
        auto output = quantized_matmul_cuda(weight_indices, input_indices, codebook);
        
        // Add bias if present
        if (bias.defined()) {
            output = output + bias;
        }
        
        return output;
    }
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto bias = saved[2];
        auto codebook = saved[3];
        auto scales = saved[4];
        auto zero_points = saved[5];
        auto depth = ctx->saved_data["depth"].toInt();
        
        // Use STE for gradients
        auto grad_input = torch::matmul(grad_outputs[0], weight);
        auto grad_weight = torch::matmul(grad_outputs[0].t(), input);
        
        torch::Tensor grad_bias;
        if (bias.defined()) {
            grad_bias = torch::sum(grad_outputs[0], 0);
        }
        
        return {grad_input, grad_weight, grad_bias, torch::Tensor(), torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

// Quantized matrix multiplication function
class QuantizedMatMulFunction : public torch::autograd::Function<QuantizedMatMulFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& input,
        const torch::Tensor& weight,
        const torch::Tensor& lookup_table
    ) {
        // Save tensors for backward pass
        ctx->save_for_backward({input, weight, lookup_table});
        
        // Perform quantized matrix multiplication
        auto output = quantized_matmul_cuda(weight, input, lookup_table);
        
        return output;
    }
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto input = saved[0];
        auto weight = saved[1];
        auto lookup_table = saved[2];
        
        // Use STE for gradients
        auto grad_input = torch::matmul(grad_outputs[0], weight);
        auto grad_weight = torch::matmul(grad_outputs[0].t(), input);
        
        return {grad_input, grad_weight, torch::Tensor()};
    }
};

// Quantized gradient function
class QuantizedGradientFunction : public torch::autograd::Function<QuantizedGradientFunction> {
public:
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        const torch::Tensor& gradients,
        const torch::Tensor& codebook,
        const torch::Tensor& scales,
        int depth
    ) {
        // Save tensors for backward pass
        ctx->save_for_backward({gradients, codebook, scales});
        ctx->saved_data["depth"] = depth;
        
        // Quantize gradients
        auto quantized_gradients = gradient_quantize_cuda(gradients, codebook, scales, depth);
        
        return quantized_gradients;
    }
    
    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs
    ) {
        auto saved = ctx->get_saved_variables();
        auto gradients = saved[0];
        auto codebook = saved[1];
        auto scales = saved[2];
        auto depth = ctx->saved_data["depth"].toInt();
        
        // Use STE for gradients
        return {grad_outputs[0], torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

// Convenience functions
torch::Tensor ste_quantize(
    const torch::Tensor& input,
    const torch::Tensor& quantized
) {
    return STEFunction::apply(input, quantized);
}

torch::Tensor quantized_linear(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const torch::Tensor& codebook,
    const torch::Tensor& scales,
    const torch::Tensor& zero_points,
    int depth
) {
    return QuantizedLinearFunction::apply(input, weight, bias, codebook, scales, zero_points, depth);
}

torch::Tensor quantized_matmul(
    const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& lookup_table
) {
    return QuantizedMatMulFunction::apply(input, weight, lookup_table);
}

torch::Tensor quantize_gradients(
    const torch::Tensor& gradients,
    const torch::Tensor& codebook,
    const torch::Tensor& scales,
    int depth
) {
    return QuantizedGradientFunction::apply(gradients, codebook, scales, depth);
}
