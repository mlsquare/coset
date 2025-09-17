#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector>

// CUDA kernel for combined quantization (encode + decode)
template<typename scalar_t>
__global__ void quantize_kernel(
    const scalar_t* __restrict__ x,           // Input vectors [batch_size, d]
    scalar_t* __restrict__ x_hat,             // Output quantized vectors [batch_size, d]
    int8_t* __restrict__ encodings,           // Intermediate encodings [batch_size, M, d]
    int32_t* __restrict__ t_values,           // Intermediate scaling counts [batch_size]
    const scalar_t* __restrict__ G,           // Generator matrix [d, d]
    const scalar_t* __restrict__ G_inv,       // Inverse generator matrix [d, d]
    const int batch_size,
    const int d,
    const int M,
    const int q,
    const scalar_t beta,
    const scalar_t alpha,
    const int max_scaling_iterations
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int batch_idx = idx / d;
    const int coord_idx = idx % d;
    
    if (batch_idx >= batch_size || coord_idx >= d) return;
    
    // Shared memory for matrices
    __shared__ scalar_t shared_G[16 * 16];
    __shared__ scalar_t shared_G_inv[16 * 16];
    
    if (threadIdx.x < d * d) {
        shared_G[threadIdx.x] = G[threadIdx.x];
        shared_G_inv[threadIdx.x] = G_inv[threadIdx.x];
    }
    __syncthreads();
    
    // Load input vector
    scalar_t x_local[16]; // Max dimension for E8
    for (int i = 0; i < d; i++) {
        x_local[i] = x[batch_idx * d + i];
    }
    
    // Apply scaling
    scalar_t x_scaled[16]; // Max dimension for E8
    for (int i = 0; i < d; i++) {
        x_scaled[i] = x_local[i] / beta;
    }
    
    // ENCODING PHASE
    scalar_t x_current[16]; // Max dimension for E8
    for (int i = 0; i < d; i++) {
        x_current[i] = x_scaled[i];
    }
    
    int t = 0;
    bool overload = true;
    
    // Try encoding with scaling if needed
    for (int scaling_iter = 0; scaling_iter < max_scaling_iterations && overload; scaling_iter++) {
        overload = false;
        
        // Encode each level
        for (int m = 0; m < M; m++) {
            // Quantize to lattice (D4 lattice quantization)
            scalar_t q_point[16]; // Max dimension for E8
            for (int i = 0; i < d; i++) {
                q_point[i] = roundf(x_current[i]);
            }
            
            // Check D4 constraint (sum must be even)
            scalar_t sum = 0.0f;
            for (int i = 0; i < d; i++) {
                sum += q_point[i];
            }
            
            // If sum is odd, adjust the coordinate farthest from integer
            if (fmodf(sum, 2.0f) != 0.0f) {
                scalar_t max_dist = 0.0f;
                int max_idx = 0;
                for (int i = 0; i < d; i++) {
                    scalar_t dist = fabsf(x_current[i] - q_point[i]);
                    if (dist > max_dist) {
                        max_dist = dist;
                        max_idx = i;
                    }
                }
                q_point[max_idx] += (q_point[max_idx] > x_current[max_idx]) ? -1.0f : 1.0f;
            }
            
            // Convert to encoding coordinates
            scalar_t Gb[16]; // Max dimension for E8
            for (int i = 0; i < d; i++) {
                Gb[i] = 0.0f;
                for (int j = 0; j < d; j++) {
                    Gb[i] += shared_G_inv[i * d + j] * q_point[j];
                }
            }
            
            // Store encoding
            for (int i = 0; i < d; i++) {
                int encoding = (int)roundf(Gb[i]) % q;
                if (encoding < 0) encoding += q;
                encodings[batch_idx * M * d + m * d + i] = (int8_t)encoding;
            }
            
            // Check for overload
            scalar_t error = 0.0f;
            for (int i = 0; i < d; i++) {
                scalar_t diff = x_current[i] - q_point[i];
                error += diff * diff;
            }
            
            if (error > 1.0f) {
                overload = true;
            }
            
            // Update for next level
            for (int i = 0; i < d; i++) {
                x_current[i] = x_current[i] - q_point[i];
            }
        }
        
        if (overload) {
            t++;
            scalar_t scale_factor = powf(2.0f, alpha);
            for (int i = 0; i < d; i++) {
                x_current[i] *= scale_factor;
            }
        }
    }
    
    // Store T value
    t_values[batch_idx] = t;
    
    // DECODING PHASE
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
        
        // Compute quantization error
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
    scalar_t scale_factor = beta * powf(2.0f, alpha * t);
    result *= scale_factor;
    
    // Store result
    x_hat[batch_idx * d + coord_idx] = result;
}

// Wrapper function for PyTorch
torch::Tensor cuda_quantize_forward(
    torch::Tensor x,
    torch::Tensor G,
    torch::Tensor G_inv,
    int q,
    int M,
    float beta,
    float alpha,
    int max_scaling_iterations
) {
    // Get dimensions
    int batch_size = x.size(0);
    int d = x.size(1);
    
    // Create intermediate tensors
    auto encodings = torch::zeros({batch_size, M, d}, torch::TensorOptions().dtype(torch::kInt8).device(x.device()));
    auto t_values = torch::zeros({batch_size}, torch::TensorOptions().dtype(torch::kInt32).device(x.device()));
    
    // Create output tensor
    auto x_hat = torch::zeros_like(x);
    
    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (batch_size * d + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "quantize_kernel", [&] {
        quantize_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            x.data_ptr<scalar_t>(),
            x_hat.data_ptr<scalar_t>(),
            encodings.data_ptr<int8_t>(),
            t_values.data_ptr<int32_t>(),
            G.data_ptr<scalar_t>(),
            G_inv.data_ptr<scalar_t>(),
            batch_size,
            d,
            M,
            q,
            static_cast<scalar_t>(beta),
            static_cast<scalar_t>(alpha),
            max_scaling_iterations
        );
    });
    
    cudaDeviceSynchronize();
    
    return x_hat;
}

// PyTorch extension binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_quantize_forward", &cuda_quantize_forward, "CUDA quantization forward");
}
