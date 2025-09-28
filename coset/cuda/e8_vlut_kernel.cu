/*
 * E8 vLUT (Value Lookup Table) Kernel for Matrix Multiplication
 * 
 * This implements efficient matrix multiplication using quantized E8 vectors
 * and lookup tables to avoid expensive quantization during computation.
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

// Device function to pack digits into index
__device__ __forceinline__ int pack_digits(int* digits, int total_digits, int Q) {
    int packed = 0;
    int base = 1;
    for (int d = 0; d < total_digits; d++) {
        packed += digits[d] * base;
        base *= Q;
    }
    return packed;
}

// Device function to unpack digits from index
__device__ __forceinline__ void unpack_digits(int packed_index, int* digits, int total_digits, int Q) {
    int remaining = packed_index;
    for (int d = 0; d < total_digits; d++) {
        digits[d] = remaining % Q;
        remaining = remaining / Q;
    }
}

// CUDA kernel for one-sided vLUT matrix multiplication
// A is quantized, B is full-precision
template<typename scalar_t>
__global__ void e8_vlut_onesided_kernel(
    const int* __restrict__ A_encoded,            // Encoded matrix A [M, N/8]
    const scalar_t* __restrict__ B,               // Full-precision matrix B [N, K]
    scalar_t* __restrict__ C,                     // Output matrix C [M, K]
    const int M,                                  // Number of rows in A
    const int N,                                  // Number of columns in A (must be multiple of 8)
    const int K,                                  // Number of columns in B
    const int Q,                                  // Quantization parameter
    const int M_levels,                           // Number of HNLQ levels
    const scalar_t* __restrict__ T_to_lat,        // Transformation matrix [8, 8]
    const scalar_t* __restrict__ G_inv,           // G_inv matrix [8, 8]
    const scalar_t* __restrict__ G,               // G matrix [8, 8]
    const scalar_t* __restrict__ vlut_table       // Pre-computed vLUT table
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = M * K;
    
    if (tid >= total_outputs) return;
    
    int m = tid / K;  // Row index in A
    int k = tid % K;  // Column index in B
    
    // Initialize output
    C[tid] = 0.0f;
    
    // Process each 8-dimensional block in A
    int N_blocks = N / 8;
    for (int block = 0; block < N_blocks; block++) {
        // Unpack encoded A vector
        int packed_index = A_encoded[m * N_blocks + block];
        int all_digits[8 * 8];  // Max 8 levels, 8 dimensions each
        unpack_digits(packed_index, all_digits, 8 * M_levels, Q);
        
        // Decode A vector for this block
        float A_decoded[8];
        for (int d = 0; d < 8; d++) {
            A_decoded[d] = 0.0f;  // Initialize reconstruction
        }
        
        for (int level = 0; level < M_levels; level++) {
            // Decode coordinates to lattice point
            int digits[8];
            for (int d = 0; d < 8; d++) {
                digits[d] = all_digits[level * 8 + d];
            }
            
            float Gb[8];
            e8_decode_coords(digits, Q, Gb, G);
            
            // Compute quantization error
            float Gb_scaled[8];
            for (int d = 0; d < 8; d++) {
                Gb_scaled[d] = Gb[d] / Q;
            }
            
            float quantized_Gb_scaled[8];
            e8_lattice_quantize(Gb_scaled, quantized_Gb_scaled);
            
            float x_i_hat[8];
            for (int d = 0; d < 8; d++) {
                x_i_hat[d] = Gb[d] - Q * quantized_Gb_scaled[d];
            }
            
            // Accumulate with appropriate weight
            float weight = powf((float)Q, (float)level);
            for (int d = 0; d < 8; d++) {
                A_decoded[d] += weight * x_i_hat[d];
            }
        }
        
        // Compute dot product with B
        for (int d = 0; d < 8; d++) {
            int B_idx = (block * 8 + d) * K + k;
            C[tid] += A_decoded[d] * B[B_idx];
        }
    }
}

// CUDA kernel for two-sided vLUT matrix multiplication
// Both A and B are quantized
template<typename scalar_t>
__global__ void e8_vlut_twosided_kernel(
    const int* __restrict__ A_encoded,            // Encoded matrix A [M, N/8]
    const int* __restrict__ B_encoded,            // Encoded matrix B [N/8, K]
    scalar_t* __restrict__ C,                     // Output matrix C [M, K]
    const int M,                                  // Number of rows in A
    const int N,                                  // Number of columns in A (must be multiple of 8)
    const int K,                                  // Number of columns in B
    const int Q,                                  // Quantization parameter
    const int M_levels,                           // Number of HNLQ levels
    const scalar_t* __restrict__ T_to_lat,        // Transformation matrix [8, 8]
    const scalar_t* __restrict__ G_inv,           // G_inv matrix [8, 8]
    const scalar_t* __restrict__ G,               // G matrix [8, 8]
    const scalar_t* __restrict__ vlut_table       // Pre-computed vLUT table
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = M * K;
    
    if (tid >= total_outputs) return;
    
    int m = tid / K;  // Row index in A
    int k = tid % K;  // Column index in B
    
    // Initialize output
    C[tid] = 0.0f;
    
    // Process each 8-dimensional block
    int N_blocks = N / 8;
    for (int block = 0; block < N_blocks; block++) {
        // Unpack encoded A vector
        int A_packed = A_encoded[m * N_blocks + block];
        int A_digits[8 * 8];  // Max 8 levels, 8 dimensions each
        unpack_digits(A_packed, A_digits, 8 * M_levels, Q);
        
        // Unpack encoded B vector
        int B_packed = B_encoded[block * K + k];
        int B_digits[8 * 8];  // Max 8 levels, 8 dimensions each
        unpack_digits(B_packed, B_digits, 8 * M_levels, Q);
        
        // For now, decode both vectors and compute dot product
        // TODO: Use vLUT table for more efficient computation
        float A_decoded[8];
        float B_decoded[8];
        
        for (int d = 0; d < 8; d++) {
            A_decoded[d] = 0.0f;
            B_decoded[d] = 0.0f;
        }
        
        for (int level = 0; level < M_levels; level++) {
            // Decode A
            int A_level_digits[8];
            for (int d = 0; d < 8; d++) {
                A_level_digits[d] = A_digits[level * 8 + d];
            }
            
            float A_Gb[8];
            e8_decode_coords(A_level_digits, Q, A_Gb, G);
            
            float A_Gb_scaled[8];
            for (int d = 0; d < 8; d++) {
                A_Gb_scaled[d] = A_Gb[d] / Q;
            }
            
            float A_quantized_scaled[8];
            e8_lattice_quantize(A_Gb_scaled, A_quantized_scaled);
            
            float A_x_i_hat[8];
            for (int d = 0; d < 8; d++) {
                A_x_i_hat[d] = A_Gb[d] - Q * A_quantized_scaled[d];
            }
            
            // Decode B
            int B_level_digits[8];
            for (int d = 0; d < 8; d++) {
                B_level_digits[d] = B_digits[level * 8 + d];
            }
            
            float B_Gb[8];
            e8_decode_coords(B_level_digits, Q, B_Gb, G);
            
            float B_Gb_scaled[8];
            for (int d = 0; d < 8; d++) {
                B_Gb_scaled[d] = B_Gb[d] / Q;
            }
            
            float B_quantized_scaled[8];
            e8_lattice_quantize(B_Gb_scaled, B_quantized_scaled);
            
            float B_x_i_hat[8];
            for (int d = 0; d < 8; d++) {
                B_x_i_hat[d] = B_Gb[d] - Q * B_quantized_scaled[d];
            }
            
            // Accumulate with appropriate weights
            float weight = powf((float)Q, (float)level);
            for (int d = 0; d < 8; d++) {
                A_decoded[d] += weight * A_x_i_hat[d];
                B_decoded[d] += weight * B_x_i_hat[d];
            }
        }
        
        // Compute dot product
        for (int d = 0; d < 8; d++) {
            C[tid] += A_decoded[d] * B_decoded[d];
        }
    }
}

/*
 * CUDA wrapper function for one-sided vLUT matrix multiplication
 */
torch::Tensor cuda_e8_vlut_onesided_matmul(
    const torch::Tensor& A_encoded,              // Encoded matrix A [M, N/8]
    const torch::Tensor& B,                      // Full-precision matrix B [N, K]
    const int Q,
    const int M_levels,
    const torch::Tensor& T_to_lat,               // Transformation matrix [8, 8]
    const torch::Tensor& G_inv,                  // G_inv matrix [8, 8]
    const torch::Tensor& G,                      // G matrix [8, 8]
    const torch::Tensor& vlut_table              // Pre-computed vLUT table
) {
    // Validate inputs
    TORCH_CHECK(A_encoded.dim() == 2, "A_encoded must be 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be 2D tensor");
    TORCH_CHECK(T_to_lat.size(0) == 8 && T_to_lat.size(1) == 8, "T_to_lat must be [8, 8] matrix");
    TORCH_CHECK(G_inv.size(0) == 8 && G_inv.size(1) == 8, "G_inv must be [8, 8] matrix");
    TORCH_CHECK(G.size(0) == 8 && G.size(1) == 8, "G must be [8, 8] matrix");
    
    int M = A_encoded.size(0);
    int N = B.size(0);
    int K = B.size(1);
    int N_blocks = N / 8;
    
    TORCH_CHECK(N % 8 == 0, "N must be multiple of 8");
    TORCH_CHECK(A_encoded.size(1) == N_blocks, "A_encoded second dimension must match N/8");
    
    // Create output tensor
    torch::Tensor C = torch::zeros({M, K}, torch::TensorOptions()
                                  .dtype(B.scalar_type())
                                  .device(A_encoded.device()));
    
    // Launch kernel
    const int threads_per_block = 256;
    const int total_outputs = M * K;
    const int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(B.scalar_type(), "e8_vlut_onesided_kernel", ([&] {
        e8_vlut_onesided_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A_encoded.data_ptr<int>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K,
            Q, M_levels,
            T_to_lat.data_ptr<scalar_t>(),
            G_inv.data_ptr<scalar_t>(),
            G.data_ptr<scalar_t>(),
            vlut_table.data_ptr<scalar_t>()
        );
    }));
    
    return C;
}

/*
 * CUDA wrapper function for two-sided vLUT matrix multiplication
 */
torch::Tensor cuda_e8_vlut_twosided_matmul(
    const torch::Tensor& A_encoded,              // Encoded matrix A [M, N/8]
    const torch::Tensor& B_encoded,              // Encoded matrix B [N/8, K]
    const int Q,
    const int M_levels,
    const torch::Tensor& T_to_lat,               // Transformation matrix [8, 8]
    const torch::Tensor& G_inv,                  // G_inv matrix [8, 8]
    const torch::Tensor& G,                      // G matrix [8, 8]
    const torch::Tensor& vlut_table              // Pre-computed vLUT table
) {
    // Validate inputs
    TORCH_CHECK(A_encoded.dim() == 2, "A_encoded must be 2D tensor");
    TORCH_CHECK(B_encoded.dim() == 2, "B_encoded must be 2D tensor");
    TORCH_CHECK(T_to_lat.size(0) == 8 && T_to_lat.size(1) == 8, "T_to_lat must be [8, 8] matrix");
    TORCH_CHECK(G_inv.size(0) == 8 && G_inv.size(1) == 8, "G_inv must be [8, 8] matrix");
    TORCH_CHECK(G.size(0) == 8 && G.size(1) == 8, "G must be [8, 8] matrix");
    
    int M = A_encoded.size(0);
    int N_blocks = A_encoded.size(1);
    int K = B_encoded.size(1);
    int N = N_blocks * 8;
    
    TORCH_CHECK(B_encoded.size(0) == N_blocks, "B_encoded first dimension must match A_encoded second dimension");
    
    // Create output tensor
    torch::Tensor C = torch::zeros({M, K}, torch::TensorOptions()
                                  .dtype(torch::kFloat32)
                                  .device(A_encoded.device()));
    
    // Launch kernel
    const int threads_per_block = 256;
    const int total_outputs = M * K;
    const int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;
    
    e8_vlut_twosided_kernel<float><<<num_blocks, threads_per_block>>>(
        A_encoded.data_ptr<int>(),
        B_encoded.data_ptr<int>(),
        C.data_ptr<float>(),
        M, N, K,
        Q, M_levels,
        T_to_lat.data_ptr<float>(),
        G_inv.data_ptr<float>(),
        G.data_ptr<float>(),
        vlut_table.data_ptr<float>()
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_e8_vlut_onesided_matmul", &cuda_e8_vlut_onesided_matmul, "CUDA E8 one-sided vLUT matrix multiplication");
    m.def("cuda_e8_vlut_twosided_matmul", &cuda_e8_vlut_twosided_matmul, "CUDA E8 two-sided vLUT matrix multiplication");
}
