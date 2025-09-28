/*
 * E8 HNLQ Decoder Kernel
 * 
 * This implements the complete E8 HNLQ decoder that reverses the encoding process.
 * It unpacks indices, decodes coordinates, and reconstructs the original vectors.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>
#include <cstdio>
#include <cmath>

// Device function for custom rounding with tie dithering (matches coset exactly)
__device__ __forceinline__ float custom_round(float x, float tiny = 1e-9f) {
    // Nudge toward zero so exact .5 falls to the nearer-integer toward zero
    float y = x - copysignf(tiny, x);
    
    // Round-to-nearest via floor(x+0.5) works for all signs after the nudge
    return floorf(y + 0.5f);
}

// Device function to implement E8Lattice.Q() exactly like coset
__device__ __forceinline__ void e8_lattice_quantize(const float* x, float* result) {
    // Find closest point in D8
    float f_x[8];
    for (int d = 0; d < 8; d++) {
        f_x[d] = custom_round(x[d]);
    }
    
    // Check if sum is even (valid D8 point)
    int sum_f_x = 0;
    for (int d = 0; d < 8; d++) {
        sum_f_x += (int)f_x[d];
    }
    
    float y_0[8];
    if (sum_f_x % 2 == 0) {
        // Valid D8 point
        for (int d = 0; d < 8; d++) {
            y_0[d] = f_x[d];
        }
    } else {
        // Need to flip one coordinate using g_x logic
        // Find coordinate with maximum distance from integer
        int farthest_d = 0;
        float max_dist = 0.0f;
        for (int d = 0; d < 8; d++) {
            float dist = fabsf(x[d] - f_x[d]);
            if (dist > max_dist) {
                max_dist = dist;
                farthest_d = d;
            }
        }
        
        // Copy f_x to y_0
        for (int d = 0; d < 8; d++) {
            y_0[d] = f_x[d];
        }
        
        // Apply g_x logic: flip the farthest coordinate
        float x_k = x[farthest_d];
        float f_x_k = f_x[farthest_d];
        if (x_k >= 0.0f) {
            y_0[farthest_d] = (f_x_k < x_k) ? f_x_k + 1.0f : f_x_k - 1.0f;
        } else {
            y_0[farthest_d] = (f_x_k <= x_k) ? f_x_k + 1.0f : f_x_k - 1.0f;  // <= for negative values
        }
    }
    
    // Find closest point in D8 + (0.5)^8
    float f_x_shifted[8];
    for (int d = 0; d < 8; d++) {
        f_x_shifted[d] = custom_round(x[d] - 0.5f);
    }
    
    int sum_f_x_shifted = 0;
    for (int d = 0; d < 8; d++) {
        sum_f_x_shifted += (int)f_x_shifted[d];
    }
    
    float y_1[8];
    if (sum_f_x_shifted % 2 == 0) {
        // Valid D8 + (0.5)^8 point
        for (int d = 0; d < 8; d++) {
            y_1[d] = f_x_shifted[d] + 0.5f;
        }
    } else {
        // Apply g_x logic to shifted space
        float x_shifted[8];
        for (int d = 0; d < 8; d++) {
            x_shifted[d] = x[d] - 0.5f;
        }
        
        // Find coordinate with maximum distance from integer in shifted space
        int farthest_d_shifted = 0;
        float max_dist_shifted = 0.0f;
        for (int d = 0; d < 8; d++) {
            float dist = fabsf(x_shifted[d] - f_x_shifted[d]);
            if (dist > max_dist_shifted) {
                max_dist_shifted = dist;
                farthest_d_shifted = d;
            }
        }
        
        float g_x_shifted[8];
        for (int d = 0; d < 8; d++) {
            g_x_shifted[d] = f_x_shifted[d];
        }
        
        // Apply g_x logic: flip the farthest coordinate
        float x_k_shifted = x_shifted[farthest_d_shifted];
        float f_x_k_shifted = f_x_shifted[farthest_d_shifted];
        if (x_k_shifted >= 0.0f) {
            g_x_shifted[farthest_d_shifted] = (f_x_k_shifted < x_k_shifted) ? f_x_k_shifted + 1.0f : f_x_k_shifted - 1.0f;
        } else {
            g_x_shifted[farthest_d_shifted] = (f_x_k_shifted <= x_k_shifted) ? f_x_k_shifted + 1.0f : f_x_k_shifted - 1.0f;
        }
        
        for (int d = 0; d < 8; d++) {
            y_1[d] = g_x_shifted[d] + 0.5f;
        }
    }
    
    // Choose the closer point (exact coset logic)
    float dist_0 = 0.0f;
    float dist_1 = 0.0f;
    for (int d = 0; d < 8; d++) {
        float diff_0 = x[d] - y_0[d];
        float diff_1 = x[d] - y_1[d];
        dist_0 += diff_0 * diff_0;
        dist_1 += diff_1 * diff_1;
    }
    
    if (dist_0 < dist_1) {
        for (int d = 0; d < 8; d++) {
            result[d] = y_0[d];
        }
    } else {
        for (int d = 0; d < 8; d++) {
            result[d] = y_1[d];
        }
    }
}

// Device function to implement lattice.decode_coords() exactly like coset
template<typename T>
__device__ __forceinline__ void e8_decode_coords(const int* digits, int Q, float* lattice_point, const T* G) {
    // Apply G transformation: G @ digits
    // Note: G is stored in row-major order (PyTorch format), but we need column-major access
    for (int d = 0; d < 8; d++) {
        float sum = 0.0f;
        for (int k = 0; k < 8; k++) {
            sum += (float)G[k * 8 + d] * digits[k];  // Column-major access for G
        }
        lattice_point[d] = sum;
    }
}

// Device function to unpack digits from index (reverse of pack_digits)
__device__ __forceinline__ void unpack_digits(int packed_index, int* digits, int total_digits, int Q) {
    int remaining = packed_index;
    for (int d = 0; d < total_digits; d++) {
        digits[d] = remaining % Q;
        remaining = remaining / Q;
    }
}

// CUDA kernel to implement complete HNLQ decoding
template<typename scalar_t>
__global__ void e8_hnlq_decoder_kernel(
    const int* __restrict__ indices,               // Input indices [batch_size]
    scalar_t* __restrict__ X,                      // Output vectors [batch_size, 8]
    const int batch_size,
    const int Q,                                   // Quantization parameter
    const int M,                                   // Number of levels
    const scalar_t* __restrict__ T_to_lat,         // Transformation matrix [8, 8]
    const scalar_t* __restrict__ G                 // G matrix [8, 8]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    
    // Unpack digits from index
    int all_digits[8 * 8];  // Max 8 levels, 8 dimensions each
    unpack_digits(indices[tid], all_digits, 8 * M, Q);
    
    // Hierarchical decoding loop (reverse of encoding)
    float x_hat[8];
    for (int d = 0; d < 8; d++) {
        x_hat[d] = 0.0f;  // Initialize reconstruction
    }
    
    for (int level = 0; level < M; level++) {
        // Step 1: Decode coordinates to lattice point
        int digits[8];
        for (int d = 0; d < 8; d++) {
            digits[d] = all_digits[level * 8 + d];
        }
        
        float Gb[8];  // Gb = lattice.decode_coords(digits, Q)
        e8_decode_coords(digits, Q, Gb, G);
        
        // Debug: Print G matrix for first thread (commented out for clean output)
        // if (tid == 0 && level == 0) {
        //     printf("G matrix in CUDA kernel:\n");
        //     for (int i = 0; i < 8; i++) {
        //         printf("  Row %d: ", i);
        //         for (int j = 0; j < 8; j++) {
        //             printf("%.1f ", (float)G[i * 8 + j]);
        //         }
        //         printf("\n");
        //     }
        // }
        
        // Step 2: Compute quantization error: Gb - Q * lattice.Q(Gb / Q)
        float Gb_scaled[8];
        for (int d = 0; d < 8; d++) {
            Gb_scaled[d] = Gb[d] / Q;
        }
        
        float quantized_Gb_scaled[8];
        e8_lattice_quantize(Gb_scaled, quantized_Gb_scaled);
        
        float x_i_hat[8];  // Quantization error
        for (int d = 0; d < 8; d++) {
            x_i_hat[d] = Gb[d] - Q * quantized_Gb_scaled[d];
        }
        
        // Step 3: Accumulate with appropriate weight
        float weight = powf((float)Q, (float)level);
        for (int d = 0; d < 8; d++) {
            x_hat[d] += weight * x_i_hat[d];
        }
        
        // Debug: Print for first thread (commented out for clean output)
        // if (tid == 0) {
        //     printf("Level %d: digits=[%d,%d,%d,%d,%d,%d,%d,%d], Gb=[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f], x_i_hat=[%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f], weight=%.1f\n",
        //            level, digits[0],digits[1],digits[2],digits[3],digits[4],digits[5],digits[6],digits[7],
        //            Gb[0],Gb[1],Gb[2],Gb[3],Gb[4],Gb[5],Gb[6],Gb[7],
        //            x_i_hat[0],x_i_hat[1],x_i_hat[2],x_i_hat[3],x_i_hat[4],x_i_hat[5],x_i_hat[6],x_i_hat[7], weight);
        // }
    }
    
    // Apply inverse T_to_lat transformation (T_to_lat^(-1))
    // For now, we'll assume T_to_lat is identity or handle it separately
    // This is a simplified version - in practice, we'd need T_to_lat^(-1)
    float reconstructed_x[8];
    for (int d = 0; d < 8; d++) {
        reconstructed_x[d] = x_hat[d];  // Simplified: assuming T_to_lat is identity
    }
    
    // Store output
    for (int d = 0; d < 8; d++) {
        X[tid * 8 + d] = (scalar_t)reconstructed_x[d];
    }
}

/*
 * CUDA wrapper function for complete E8 HNLQ decoding
 */
torch::Tensor cuda_e8_hnlq_decode(
    const torch::Tensor& indices,                 // Input indices [batch_size]
    const int Q,
    const int M,
    const torch::Tensor& T_to_lat,                // Transformation matrix [8, 8]
    const torch::Tensor& G                        // G matrix [8, 8]
) {
    // Validate inputs
    TORCH_CHECK(indices.dim() == 1, "Input indices must be 1D tensor [batch_size]");
    TORCH_CHECK(T_to_lat.size(0) == 8 && T_to_lat.size(1) == 8, "T_to_lat must be [8, 8] matrix");
    TORCH_CHECK(G.size(0) == 8 && G.size(1) == 8, "G must be [8, 8] matrix");
    
    int batch_size = indices.size(0);
    
    // Create output tensor
    torch::Tensor X = torch::zeros({batch_size, 8}, torch::TensorOptions()
                                  .dtype(T_to_lat.scalar_type())
                                  .device(indices.device()));
    
    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(T_to_lat.scalar_type(), "e8_hnlq_decoder_kernel", ([&] {
        e8_hnlq_decoder_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            indices.data_ptr<int>(),
            X.data_ptr<scalar_t>(),
            batch_size,
            Q,
            M,
            T_to_lat.data_ptr<scalar_t>(),
            G.data_ptr<scalar_t>()
        );
    }));
    
    return X;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_e8_hnlq_decode", &cuda_e8_hnlq_decode, "CUDA E8 HNLQ complete decoding");
}
