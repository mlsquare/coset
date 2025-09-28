/*
 * Step 3: E8 HNLQ Encoder Kernel
 * 
 * This implements the complete hierarchical encoding loop with multiple levels.
 * Combines E8 lattice quantization + coordinate encoding + hierarchical loop.
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
        int farthest_d = 0;
        float max_dist = 0.0f;
        for (int d = 0; d < 8; d++) {
            float dist = fabsf(x_shifted[d] - f_x_shifted[d]);
            if (dist > max_dist) {
                max_dist = dist;
                farthest_d = d;
            }
        }
        
        // Copy f_x_shifted to y_1
        for (int d = 0; d < 8; d++) {
            y_1[d] = f_x_shifted[d] + 0.5f;
        }
        
        // Apply g_x logic to shifted space
        float x_k_shifted = x_shifted[farthest_d];
        float f_x_k_shifted = f_x_shifted[farthest_d];
        if (x_k_shifted >= 0.0f) {
            y_1[farthest_d] = (f_x_k_shifted < x_k_shifted) ? f_x_k_shifted + 1.5f : f_x_k_shifted - 0.5f;
        } else {
            y_1[farthest_d] = (f_x_k_shifted <= x_k_shifted) ? f_x_k_shifted + 1.5f : f_x_k_shifted - 0.5f;  // <= for negative values
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

// Device function to implement lattice.encode_coords() exactly like coset
template<typename T>
__device__ __forceinline__ void e8_encode_coords(const float* lattice_point, int Q, int* digits, const T* G_inv) {
    // Apply G_inv transformation: round(G_inv @ lattice_point) % Q
    // Note: G_inv is stored in row-major order (PyTorch format)
    for (int d = 0; d < 8; d++) {
        float transformed = 0.0f;
        for (int k = 0; k < 8; k++) {
            transformed += (float)G_inv[k * 8 + d] * lattice_point[k];  // Column-major access
        }
        
        float rounded = custom_round(transformed);
        int digit = ((int)rounded) % Q;
        if (digit < 0) digit += Q;
        digits[d] = digit;
    }
}

// Device function to implement index packing (simple level-packed format)
__device__ __forceinline__ int pack_digits(int* digits, int D, int Q) {
    int packed = 0;
    int base = 1;
    for (int d = 0; d < D; d++) {
        packed += digits[d] * base;
        base *= Q;
    }
    return packed;
}

// CUDA kernel to implement complete HNLQ encoding
template<typename scalar_t>
__global__ void e8_hnlq_encoder_kernel(
    const scalar_t* __restrict__ X,               // Input vectors [batch_size, 8]
    int* __restrict__ indices,                    // Output indices [batch_size]
    const int batch_size,
    const int Q,                                  // Quantization parameter
    const int M,                                  // Number of levels
    const scalar_t* __restrict__ T_to_lat,        // Transformation matrix [8, 8]
    const scalar_t* __restrict__ G_inv            // G_inv matrix [8, 8]
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    
    // Load input vector
    float x[8];
    for (int d = 0; d < 8; d++) {
        x[d] = (float)X[tid * 8 + d];
    }
    
    // Apply T_to_lat transformation
    float transformed_x[8];
    for (int d = 0; d < 8; d++) {
        float sum = 0.0f;
        for (int k = 0; k < 8; k++) {
            sum += T_to_lat[d * 8 + k] * x[k];  // Row-major access for T_to_lat
        }
        transformed_x[d] = sum;
    }
    
    // Hierarchical encoding loop
    float current_x[8];
    for (int d = 0; d < 8; d++) {
        current_x[d] = transformed_x[d];
    }
    
    int all_digits[8 * 8];  // Max 8 levels, 8 dimensions each
    
    for (int level = 0; level < M; level++) {
        // Step 1: E8 lattice quantization
        float lattice_point[8];
        e8_lattice_quantize(current_x, lattice_point);
        
        // Step 2: Coordinate encoding
        int digits[8];
        e8_encode_coords(lattice_point, Q, digits, G_inv);
        
        // Store digits for this level
        for (int d = 0; d < 8; d++) {
            all_digits[level * 8 + d] = digits[d];
        }
        
        // Step 3: Scale down for next level (hierarchical encoding)
        if (level < M - 1) {
            // Scale down the quantized lattice point by Q for next level
            for (int d = 0; d < 8; d++) {
                current_x[d] = lattice_point[d] / Q;
            }
        }
    }
    
    // Pack all digits into final index
    int packed_index = pack_digits(all_digits, 8 * M, Q);
    indices[tid] = packed_index;
    
    // Debug: Print digits for first thread (commented out for clean output)
    // if (tid == 0) {
    //     printf("CUDA digits for test vector [1,1,1,1,1,1,1,1]:\n");
    //     for (int level = 0; level < M; level++) {
    //         printf("  Level %d: ", level);
    //         for (int d = 0; d < 8; d++) {
    //             printf("%d ", all_digits[level * 8 + d]);
    //         }
    //         printf("\n");
    //     }
    //     printf("Packed index: %d\n", packed_index);
    // }
}

/*
 * CUDA wrapper function for complete E8 HNLQ encoding
 */
torch::Tensor cuda_e8_hnlq_encode(
    const torch::Tensor& X,                       // Input vectors [batch_size, 8]
    const int Q,
    const int M,
    const torch::Tensor& T_to_lat,                // Transformation matrix [8, 8]
    const torch::Tensor& G_inv                    // G_inv matrix [8, 8]
) {
    // Validate inputs
    TORCH_CHECK(X.dim() == 2, "Input X must be 2D tensor [batch_size, 8]");
    TORCH_CHECK(X.size(1) == 8, "Input X last dimension must be 8");
    TORCH_CHECK(T_to_lat.size(0) == 8 && T_to_lat.size(1) == 8, "T_to_lat must be [8, 8] matrix");
    TORCH_CHECK(G_inv.size(0) == 8 && G_inv.size(1) == 8, "G_inv must be [8, 8] matrix");
    
    int batch_size = X.size(0);
    
    // Create output tensor
    torch::Tensor indices = torch::zeros({batch_size}, torch::TensorOptions()
                                        .dtype(torch::kInt32)
                                        .device(X.device()));
    
    // Launch kernel
    const int threads_per_block = 256;
    const int num_blocks = (batch_size + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "e8_hnlq_encoder_kernel", ([&] {
        e8_hnlq_encoder_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            X.data_ptr<scalar_t>(),
            indices.data_ptr<int>(),
            batch_size,
            Q,
            M,
            T_to_lat.data_ptr<scalar_t>(),
            G_inv.data_ptr<scalar_t>()
        );
    }));
    
    return indices;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_e8_hnlq_encode", &cuda_e8_hnlq_encode, "CUDA E8 HNLQ complete encoding");
}
