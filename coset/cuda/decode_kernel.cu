#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

// CUDA kernel for hierarchical decoding
template<typename scalar_t>
__global__ void decode_kernel(
    const int8_t* __restrict__ encodings,    // Input encodings [batch_size, M, d]
    const int32_t* __restrict__ t_values,    // Input scaling counts [batch_size]
    scalar_t* __restrict__ x_hat,            // Output decoded vectors [batch_size, d]
    const scalar_t* __restrict__ G,          // Generator matrix [d, d]
    const int batch_size,
    const int d,
    const int M,
    const int q,
    const scalar_t beta,
    const scalar_t alpha
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / d;
    const int coord_idx = idx % d;
    
    if (batch_idx >= batch_size || coord_idx >= d) return;
    
    // Shared memory for generator matrix
    __shared__ scalar_t shared_G[16 * 16]; // Max 16x16 for E8
    if (threadIdx.x < d * d) {
        shared_G[threadIdx.x] = G[threadIdx.x];
    }
    __syncthreads();
    
    // Initialize output
    scalar_t result = 0.0f;
    
    // Decode each level
    for (int m = 0; m < M; m++) {
        // Load encoding for this level
        int8_t b_m[16]; // Max dimension for E8
        for (int i = 0; i < d; i++) {
            b_m[i] = encodings[batch_idx * M * d + m * d + i];
        }
        
        // Convert encoding coordinates to lattice points
        scalar_t Gb[16]; // Max dimension for E8
        for (int i = 0; i < d; i++) {
            Gb[i] = 0.0f;
            for (int j = 0; j < d; j++) {
                Gb[i] += shared_G[i * d + j] * (scalar_t)b_m[j];
            }
        }
        
        // Compute quantization error (simplified)
        scalar_t q_points[16]; // Max dimension for E8
        for (int i = 0; i < d; i++) {
            q_points[i] = roundf(Gb[i] / q);
        }
        
        scalar_t x_m_hat = Gb[coord_idx] - q * q_points[coord_idx];
        
        // Accumulate with appropriate weight
        scalar_t weight = powf((scalar_t)q, (scalar_t)m);
        result += weight * x_m_hat;
    }
    
    // Apply scaling compensation
    int t = t_values[batch_idx];
    scalar_t scale_factor = beta * powf(2.0f, alpha * t);
    result *= scale_factor;
    
    // Store result
    x_hat[batch_idx * d + coord_idx] = result;
}

// Wrapper function for PyTorch
torch::Tensor cuda_decode_forward(
    torch::Tensor encodings,
    torch::Tensor t_values,
    torch::Tensor G,
    int q,
    int M,
    float beta,
    float alpha
) {
    // Get dimensions
    int batch_size = encodings.size(0);
    int d = encodings.size(2);
    
    // Create output tensor
    auto x_hat = torch::zeros({batch_size, d}, torch::TensorOptions().dtype(encodings.scalar_type()).device(encodings.device()));
    
    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (batch_size * d + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(x_hat.scalar_type(), "decode_kernel", [&] {
        decode_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            encodings.data_ptr<int8_t>(),
            t_values.data_ptr<int32_t>(),
            x_hat.data_ptr<scalar_t>(),
            G.data_ptr<scalar_t>(),
            batch_size,
            d,
            M,
            q,
            static_cast<scalar_t>(beta),
            static_cast<scalar_t>(alpha)
        );
    });
    
    cudaDeviceSynchronize();
    
    return x_hat;
}

// PyTorch extension binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_decode_forward", &cuda_decode_forward, "CUDA decoding forward");
}
