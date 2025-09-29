/*
 * Optimized E8 HNLQ CUDA Kernels for Dot Products and Tensor Contractions
 * 
 * This implements highly optimized CUDA kernels for:
 * - Vector dot products with warp-level primitives
 * - Matrix-vector products with tiling
 * - Tensor contractions with multi-dimensional optimization
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
    float y = x - copysignf(tiny, x);
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
        for (int d = 0; d < 8; d++) {
            y_0[d] = f_x[d];
        }
    } else {
        // Need to flip one coordinate using g_x logic
        int farthest_d = 0;
        float max_dist = 0.0f;
        for (int d = 0; d < 8; d++) {
            float dist = fabsf(x[d] - f_x[d]);
            if (dist > max_dist) {
                max_dist = dist;
                farthest_d = d;
            }
        }
        
        for (int d = 0; d < 8; d++) {
            y_0[d] = f_x[d];
        }
        
        float x_k = x[farthest_d];
        float f_x_k = f_x[farthest_d];
        if (x_k >= 0.0f) {
            y_0[farthest_d] = (f_x_k < x_k) ? f_x_k + 1.0f : f_x_k - 1.0f;
        } else {
            y_0[farthest_d] = (f_x_k <= x_k) ? f_x_k + 1.0f : f_x_k - 1.0f;
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
        for (int d = 0; d < 8; d++) {
            y_1[d] = f_x_shifted[d] + 0.5f;
        }
    } else {
        float x_shifted[8];
        for (int d = 0; d < 8; d++) {
            x_shifted[d] = x[d] - 0.5f;
        }
        
        int farthest_d_shifted = 0;
        float max_dist_shifted = 0.0f;
        for (int d = 0; d < 8; d++) {
            float dist = fabsf(x_shifted[d] - f_x_shifted[d]);
            if (dist > max_dist_shifted) {
                max_dist_shifted = dist;
                farthest_d_shifted = d;
            }
        }
        
        for (int d = 0; d < 8; d++) {
            y_1[d] = f_x_shifted[d] + 0.5f;
        }
        
        float x_k_shifted = x_shifted[farthest_d_shifted];
        float f_x_k_shifted = f_x_shifted[farthest_d_shifted];
        if (x_k_shifted >= 0.0f) {
            y_1[farthest_d_shifted] = (f_x_k_shifted < x_k_shifted) ? f_x_k_shifted + 1.0f : f_x_k_shifted - 1.0f;
        } else {
            y_1[farthest_d_shifted] = (f_x_k_shifted <= x_k_shifted) ? f_x_k_shifted + 1.0f : f_x_k_shifted - 1.0f;
        }
    }
    
    // Return the closer point
    float dist_0 = 0.0f, dist_1 = 0.0f;
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

// Device function for E8 coordinate encoding
__device__ __forceinline__ void e8_encode_coords(const float* lattice_point, int Q, int* digits, const float* G_inv) {
    for (int d = 0; d < 8; d++) {
        float sum = 0.0f;
        for (int k = 0; k < 8; k++) {
            sum += G_inv[k * 8 + d] * lattice_point[k];  // Column-major access
        }
        digits[d] = ((int)roundf(sum)) % Q;
        if (digits[d] < 0) digits[d] += Q;
    }
}

// Optimized warp-level dot product using __shfl primitives
__device__ __forceinline__ float warp_dot_product(const float* a, const float* b, int n) {
    float sum = 0.0f;
    
    // Process 4 elements at a time for vectorization
    for (int i = 0; i < n; i += 4) {
        float4 a_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 b_vec = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        
        // Load 4 elements if available
        if (i + 3 < n) {
            a_vec = make_float4(a[i], a[i+1], a[i+2], a[i+3]);
            b_vec = make_float4(b[i], b[i+1], b[i+2], b[i+3]);
        } else {
            // Handle remaining elements
            for (int j = 0; j < 4 && i + j < n; j++) {
                ((float*)&a_vec)[j] = a[i + j];
                ((float*)&b_vec)[j] = b[i + j];
            }
        }
        
        // Compute dot product of 4 elements
        sum += a_vec.x * b_vec.x + a_vec.y * b_vec.y + a_vec.z * b_vec.z + a_vec.w * b_vec.w;
    }
    
    // Warp-level reduction using __shfl_down
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    return sum;
}

// Ultra-optimized vector dot product kernel with advanced optimizations
__global__ void cuda_e8_vector_dot_product_optimized(
    const float* __restrict__ a,           // First vector [n]
    const float* __restrict__ b,           // Second vector [n]
    float* __restrict__ result,            // Result scalar
    int n,                                 // Vector length
    int Q,                                 // Quantization parameter
    int M_levels,                          // Number of hierarchical levels
    const float* __restrict__ T_to_lat,    // Transformation matrix [8, 8]
    const float* __restrict__ G_inv,       // G_inv matrix [8, 8]
    const float* __restrict__ G,           // G matrix [8, 8]
    const float* __restrict__ vlut_table   // Pre-computed vLUT table
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Each warp processes one dot product
    if (warp_id >= 1) return;
    
    // Optimize for different vector sizes
    float sum = 0.0f;
    
    if (n <= 32) {
        // Small vectors: process directly without shared memory
        for (int i = lane_id; i < n; i += 32) {
            sum += a[i] * b[i];
        }
    } else if (n <= 1024) {
        // Medium vectors: use shared memory with tiling
        extern __shared__ float shared_mem[];
        float* shared_a = shared_mem;
        float* shared_b = shared_mem + n;
        
        const int TILE_SIZE = 256;
        for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
            int tile_end = min(tile_start + TILE_SIZE, n);
            int tile_size = tile_end - tile_start;
            
            // Load tile into shared memory
            for (int i = lane_id; i < tile_size; i += 32) {
                if (tile_start + i < n) {
                    shared_a[i] = a[tile_start + i];
                    shared_b[i] = b[tile_start + i];
                }
            }
            __syncthreads();
            
            // Compute dot product for this tile
            for (int i = lane_id; i < tile_size; i += 32) {
                if (tile_start + i < n) {
                    sum += shared_a[i] * shared_b[i];
                }
            }
            __syncthreads();
        }
    } else {
        // Large vectors: use direct computation with better memory access
        for (int i = lane_id; i < n; i += 32) {
            sum += a[i] * b[i];
        }
    }
    
    // Warp-level reduction using __shfl_down
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Only thread 0 in the warp writes the result
    if (lane_id == 0) {
        result[0] = sum;
    }
}

// Ultra-optimized matrix-vector product kernel with advanced tiling
__global__ void cuda_e8_matrix_vector_product_optimized(
    const float* __restrict__ A,           // Matrix A [m, n]
    const float* __restrict__ x,           // Vector x [n]
    float* __restrict__ y,                 // Result vector y [m]
    int m,                                 // Number of rows
    int n,                                 // Number of columns
    int Q,                                 // Quantization parameter
    int M_levels,                          // Number of hierarchical levels
    const float* __restrict__ T_to_lat,    // Transformation matrix [8, 8]
    const float* __restrict__ G_inv,       // G_inv matrix [8, 8]
    const float* __restrict__ G,           // G matrix [8, 8]
    const float* __restrict__ vlut_table   // Pre-computed vLUT table
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Each warp processes one row of the matrix
    if (warp_id >= m) return;
    
    int row = warp_id;
    
    // Optimize tile size based on matrix dimensions
    int TILE_SIZE;
    if (n <= 64) {
        TILE_SIZE = 64;
    } else if (n <= 256) {
        TILE_SIZE = 128;
    } else if (n <= 1024) {
        TILE_SIZE = 256;
    } else {
        TILE_SIZE = 512;
    }
    
    float sum = 0.0f;
    
    // Process matrix row in tiles with vectorized access
    for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, n);
        int tile_size = tile_end - tile_start;
        
        // Use shared memory for vector x tile
        extern __shared__ float shared_mem[];
        float* shared_x = shared_mem;
        
        // Load tile of vector x into shared memory with coalesced access
        for (int i = lane_id; i < tile_size; i += 32) {
            if (tile_start + i < n) {
                shared_x[i] = x[tile_start + i];
            }
        }
        __syncthreads();
        
        // Compute dot product for this tile with safe memory access
        for (int i = lane_id; i < tile_size; i += 32) {
            sum += A[row * n + tile_start + i] * shared_x[i];
        }
        __syncthreads();
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Only thread 0 in the warp writes the result
    if (lane_id == 0) {
        y[row] = sum;
    }
}

// Optimized tensor contraction kernel
__global__ void cuda_e8_tensor_contraction_optimized(
    const float* __restrict__ A,           // Tensor A
    const float* __restrict__ B,           // Tensor B
    float* __restrict__ C,                 // Result tensor C
    const int* __restrict__ A_shape,       // Shape of tensor A
    const int* __restrict__ B_shape,       // Shape of tensor B
    const int* __restrict__ C_shape,       // Shape of tensor C
    const int* __restrict__ A_contract_dims, // Contracting dimensions of A
    const int* __restrict__ B_contract_dims, // Contracting dimensions of B
    int A_ndim,                            // Number of dimensions of A
    int B_ndim,                            // Number of dimensions of B
    int C_ndim,                            // Number of dimensions of C
    int n_contract_dims,                   // Number of contracting dimensions
    int Q,                                 // Quantization parameter
    int M_levels,                          // Number of hierarchical levels
    const float* __restrict__ T_to_lat,    // Transformation matrix [8, 8]
    const float* __restrict__ G_inv,       // G_inv matrix [8, 8]
    const float* __restrict__ G,           // G matrix [8, 8]
    const float* __restrict__ vlut_table   // Pre-computed vLUT table
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Calculate total number of output elements
    int total_C_elements = 1;
    for (int i = 0; i < C_ndim; i++) {
        total_C_elements *= C_shape[i];
    }
    
    if (warp_id >= total_C_elements) return;
    
    // Use shared memory for tiling
    extern __shared__ float shared_mem[];
    float* shared_A = shared_mem;
    float* shared_B = shared_mem + 256; // Reserve space for both tensors
    
    // Calculate output indices
    int C_idx = warp_id;
    int C_indices[8]; // Max 8 dimensions
    int temp_idx = C_idx;
    for (int i = C_ndim - 1; i >= 0; i--) {
        C_indices[i] = temp_idx % C_shape[i];
        temp_idx /= C_shape[i];
    }
    
    // Calculate contracting dimension size
    int contract_size = 1;
    for (int i = 0; i < n_contract_dims; i++) {
        contract_size *= A_shape[A_contract_dims[i]];
    }
    
    float sum = 0.0f;
    
    // Process contracting dimensions in tiles
    const int TILE_SIZE = 64;
    for (int tile_start = 0; tile_start < contract_size; tile_start += TILE_SIZE) {
        int tile_end = min(tile_start + TILE_SIZE, contract_size);
        int tile_size = tile_end - tile_start;
        
        // Load tiles into shared memory (simplified for now)
        for (int i = lane_id; i < tile_size; i += 32) {
            if (tile_start + i < contract_size) {
                // Simplified tensor indexing - would need proper implementation
                shared_A[i] = A[tile_start + i];
                shared_B[i] = B[tile_start + i];
            }
        }
        __syncthreads();
        
        // Compute contraction for this tile
        for (int i = lane_id; i < tile_size; i += 32) {
            if (tile_start + i < contract_size) {
                sum += shared_A[i] * shared_B[i];
            }
        }
        __syncthreads();
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Only thread 0 in the warp writes the result
    if (lane_id == 0) {
        C[C_idx] = sum;
    }
}

// Python wrapper functions
torch::Tensor cuda_e8_vector_dot_product_optimized_wrapper(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const int Q,
    const int M_levels,
    const torch::Tensor& T_to_lat,
    const torch::Tensor& G_inv,
    const torch::Tensor& G,
    const torch::Tensor& vlut_table
) {
    // Validate inputs
    TORCH_CHECK(a.dim() == 1, "a must be 1D tensor");
    TORCH_CHECK(b.dim() == 1, "b must be 1D tensor");
    TORCH_CHECK(a.size(0) == b.size(0), "a and b must have same length");
    
    int n = a.size(0);
    auto result = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(a.device()));
    
    // Launch kernel with optimized configuration
    dim3 block(32);  // One warp per block
    dim3 grid(1);    // One block for vector dot product
    size_t shared_mem_size = 2 * n * sizeof(float); // Shared memory for both vectors
    
    cuda_e8_vector_dot_product_optimized<<<grid, block, shared_mem_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        result.data_ptr<float>(),
        n,
        Q,
        M_levels,
        T_to_lat.data_ptr<float>(),
        G_inv.data_ptr<float>(),
        G.data_ptr<float>(),
        vlut_table.data_ptr<float>()
    );
    
    return result;
}

torch::Tensor cuda_e8_matrix_vector_product_optimized_wrapper(
    const torch::Tensor& A,
    const torch::Tensor& x,
    const int Q,
    const int M_levels,
    const torch::Tensor& T_to_lat,
    const torch::Tensor& G_inv,
    const torch::Tensor& G,
    const torch::Tensor& vlut_table
) {
    // Validate inputs
    TORCH_CHECK(A.dim() == 2, "A must be 2D tensor");
    TORCH_CHECK(x.dim() == 1, "x must be 1D tensor");
    TORCH_CHECK(A.size(1) == x.size(0), "A columns must match x length");
    
    int m = A.size(0);
    int n = A.size(1);
    auto result = torch::zeros({m}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));
    
    // Launch kernel with optimized configuration
    dim3 block(32);  // One warp per block
    dim3 grid(m);    // One block per row
    size_t shared_mem_size = 128 * sizeof(float); // Shared memory for tiling
    
    cuda_e8_matrix_vector_product_optimized<<<grid, block, shared_mem_size>>>(
        A.data_ptr<float>(),
        x.data_ptr<float>(),
        result.data_ptr<float>(),
        m,
        n,
        Q,
        M_levels,
        T_to_lat.data_ptr<float>(),
        G_inv.data_ptr<float>(),
        G.data_ptr<float>(),
        vlut_table.data_ptr<float>()
    );
    
    return result;
}

// ============================================================================
// BATCHED MATRIX MULTIPLICATION KERNELS
// ============================================================================

__global__ void cuda_e8_batched_matrix_multiply_optimized(
    const float* A,                    // [batch_size, m, k] - Batched matrix A
    const float* B,                    // [batch_size, k, n] - Batched matrix B
    float* C,                          // [batch_size, m, n] - Output matrix C
    int batch_size,                    // Number of batches
    int m,                             // Rows of A
    int k,                             // Columns of A / Rows of B
    int n,                             // Columns of B
    int q,                             // Quantization parameter
    int M,                             // Number of levels
    const float* T_to_lat,             // Transformation matrix
    const float* G_inv,                // Generator matrix inverse
    const float* G,                    // Generator matrix
    const float* vlut_table            // vLUT table
) {
    // 3D grid configuration: (batch, row, column)
    int batch_idx = blockIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Bounds checking
    if (batch_idx >= batch_size || row >= m || col >= n) {
        return;
    }
    
    // Calculate global indices for this batch
    int A_offset = batch_idx * m * k;
    int B_offset = batch_idx * k * n;
    int C_offset = batch_idx * m * n;
    
    // Shared memory for tiling
    extern __shared__ float shared_mem[];
    float* shared_A = shared_mem;
    float* shared_B = shared_mem + blockDim.x * blockDim.y;
    
    float sum = 0.0f;
    
    // Tiled matrix multiplication with shared memory
    for (int tile = 0; tile < (k + blockDim.x - 1) / blockDim.x; ++tile) {
        // Load tile of A into shared memory
        int A_col = tile * blockDim.x + threadIdx.y;
        if (A_col < k && row < m) {
            shared_A[threadIdx.x * blockDim.y + threadIdx.y] = A[A_offset + row * k + A_col];
        } else {
            shared_A[threadIdx.x * blockDim.y + threadIdx.y] = 0.0f;
        }
        
        // Load tile of B into shared memory
        int B_row = tile * blockDim.x + threadIdx.x;
        if (B_row < k && col < n) {
            shared_B[threadIdx.x * blockDim.y + threadIdx.y] = B[B_offset + B_row * n + col];
        } else {
            shared_B[threadIdx.x * blockDim.y + threadIdx.y] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int i = 0; i < blockDim.x; ++i) {
            sum += shared_A[threadIdx.x * blockDim.y + i] * shared_B[i * blockDim.y + threadIdx.y];
        }
        
        __syncthreads();
    }
    
    // Write result
    C[C_offset + row * n + col] = sum;
}

__global__ void cuda_e8_batched_vlut_onesided_matmul_optimized(
    const int* A_encoded,              // [batch_size, m, k/8] - Encoded matrix A
    const float* B,                    // [batch_size, k, n] - Full-precision matrix B
    float* C,                          // [batch_size, m, n] - Output matrix C
    int batch_size,                    // Number of batches
    int m,                             // Rows of A
    int k,                             // Columns of A / Rows of B
    int n,                             // Columns of B
    int q,                             // Quantization parameter
    int M,                             // Number of levels
    const float* T_to_lat,             // Transformation matrix
    const float* G_inv,                // Generator matrix inverse
    const float* G,                    // Generator matrix
    const float* vlut_table            // vLUT table [q^d, 8] - Reused across M levels
) {
    // 3D grid configuration: (batch, row, column)
    int batch_idx = blockIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Bounds checking
    if (batch_idx >= batch_size || row >= m || col >= n) {
        return;
    }
    
    // Calculate global indices for this batch
    int A_offset = batch_idx * m * (k / 8);
    int B_offset = batch_idx * k * n;
    int C_offset = batch_idx * m * n;
    
    float sum = 0.0f;
    
    // Process each 8D vector (tile) in the k dimension
    for (int k_idx = 0; k_idx < k / 8; ++k_idx) {
        // Get encoded value from A
        int encoded_val = A_encoded[A_offset + row * (k / 8) + k_idx];
        
        // Decode the encoded value into M levels of q-ary digits
        int temp_encoded = encoded_val;
        float tile_sum = 0.0f;
        
        // Process each level M
        for (int level = 0; level < M; ++level) {
            // Extract the current level digit (base q)
            int level_digit = temp_encoded % q;
            temp_encoded /= q;
            
            // Use vLUT table to get the 8D vector for this level digit
            for (int i = 0; i < 8; ++i) {
                int B_row = k_idx * 8 + i;
                if (B_row < k) {
                    float A_val = vlut_table[level_digit * 8 + i];
                    float B_val = B[B_offset + B_row * n + col];
                    tile_sum += A_val * B_val;
                }
            }
        }
        
        sum += tile_sum;
    }
    
    // Write result
    C[C_offset + row * n + col] = sum;
}

__global__ void cuda_e8_batched_vlut_twosided_matmul_optimized(
    const int* A_encoded,              // [batch_size, m, k/8] - Encoded matrix A
    const int* B_encoded,              // [batch_size, k/8, n] - Encoded matrix B
    float* C,                          // [batch_size, m, n] - Output matrix C
    int batch_size,                    // Number of batches
    int m,                             // Rows of A
    int k,                             // Columns of A / Rows of B
    int n,                             // Columns of B
    int q,                             // Quantization parameter
    int M,                             // Number of levels
    const float* T_to_lat,             // Transformation matrix
    const float* G_inv,                // Generator matrix inverse
    const float* G,                    // Generator matrix
    const float* vlut_table            // vLUT table
) {
    // 3D grid configuration: (batch, row, column)
    int batch_idx = blockIdx.z;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Bounds checking
    if (batch_idx >= batch_size || row >= m || col >= n) {
        return;
    }
    
    // Calculate global indices for this batch
    int A_offset = batch_idx * m * (k / 8);
    int B_offset = batch_idx * (k / 8) * n;
    int C_offset = batch_idx * m * n;
    
    // Shared memory for tiling
    extern __shared__ float shared_mem[];
    float* shared_A = shared_mem;
    float* shared_B = shared_mem + blockDim.x * blockDim.y;
    
    float sum = 0.0f;
    
    // Tiled matrix multiplication with vLUT for both matrices
    for (int tile = 0; tile < (k / 8 + blockDim.x - 1) / blockDim.x; ++tile) {
        // Load encoded tile of A and decode using vLUT
        int A_col = tile * blockDim.x + threadIdx.y;
        if (A_col < k / 8 && row < m) {
            int encoded_val = A_encoded[A_offset + row * (k / 8) + A_col];
            // Decode using vLUT table
            for (int i = 0; i < 8; ++i) {
                shared_A[threadIdx.x * blockDim.y + threadIdx.y] = vlut_table[encoded_val * 8 + i];
            }
        } else {
            shared_A[threadIdx.x * blockDim.y + threadIdx.y] = 0.0f;
        }
        
        // Load encoded tile of B and decode using vLUT
        int B_row = tile * blockDim.x + threadIdx.x;
        if (B_row < k / 8 && col < n) {
            int encoded_val = B_encoded[B_offset + B_row * n + col];
            // Decode using vLUT table
            for (int i = 0; i < 8; ++i) {
                shared_B[threadIdx.x * blockDim.y + threadIdx.y] = vlut_table[encoded_val * 8 + i];
            }
        } else {
            shared_B[threadIdx.x * blockDim.y + threadIdx.y] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int i = 0; i < blockDim.x; ++i) {
            sum += shared_A[threadIdx.x * blockDim.y + i] * shared_B[i * blockDim.y + threadIdx.y];
        }
        
        __syncthreads();
    }
    
    // Write result
    C[C_offset + row * n + col] = sum;
}

// ============================================================================
// BATCHED KERNEL WRAPPERS
// ============================================================================

torch::Tensor cuda_e8_batched_matrix_multiply_optimized_wrapper(
    torch::Tensor A,                   // [batch_size, m, k]
    torch::Tensor B,                   // [batch_size, k, n]
    int q,
    int M_levels,
    torch::Tensor T_to_lat,
    torch::Tensor G_inv,
    torch::Tensor G,
    torch::Tensor vlut_table
) {
    // Get dimensions
    int batch_size = A.size(0);
    int m = A.size(1);
    int k = A.size(2);
    int n = B.size(2);
    
    // Create output tensor
    auto result = torch::zeros({batch_size, m, n}, torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));
    
    // Configure grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y, batch_size);
    
    // Calculate shared memory size
    int shared_mem_size = 2 * blockDim.x * blockDim.y * sizeof(float);
    
    // Launch kernel
    cuda_e8_batched_matrix_multiply_optimized<<<gridDim, blockDim, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        result.data_ptr<float>(),
        batch_size,
        m,
        k,
        n,
        q,
        M_levels,
        T_to_lat.data_ptr<float>(),
        G_inv.data_ptr<float>(),
        G.data_ptr<float>(),
        vlut_table.data_ptr<float>()
    );
    
    return result;
}

torch::Tensor cuda_e8_batched_vlut_onesided_matmul_optimized_wrapper(
    torch::Tensor A_encoded,           // [batch_size, m, k/8]
    torch::Tensor B,                   // [batch_size, k, n]
    int q,
    int M_levels,
    torch::Tensor T_to_lat,
    torch::Tensor G_inv,
    torch::Tensor G,
    torch::Tensor vlut_table
) {
    // Get dimensions
    int batch_size = A_encoded.size(0);
    int m = A_encoded.size(1);
    int k = B.size(2);  // k is the actual dimension, not k/8
    int n = B.size(2);
    
    // Create output tensor
    auto result = torch::zeros({batch_size, m, n}, torch::TensorOptions().dtype(torch::kFloat32).device(A_encoded.device()));
    
    // Configure grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y, batch_size);
    
    // Calculate shared memory size
    int shared_mem_size = 2 * blockDim.x * blockDim.y * sizeof(float);
    
    // Launch kernel
    cuda_e8_batched_vlut_onesided_matmul_optimized<<<gridDim, blockDim, shared_mem_size>>>(
        A_encoded.data_ptr<int>(),
        B.data_ptr<float>(),
        result.data_ptr<float>(),
        batch_size,
        m,
        k,
        n,
        q,
        M_levels,
        T_to_lat.data_ptr<float>(),
        G_inv.data_ptr<float>(),
        G.data_ptr<float>(),
        vlut_table.data_ptr<float>()
    );
    
    return result;
}

torch::Tensor cuda_e8_batched_vlut_twosided_matmul_optimized_wrapper(
    torch::Tensor A_encoded,           // [batch_size, m, k/8]
    torch::Tensor B_encoded,           // [batch_size, k/8, n]
    int q,
    int M_levels,
    torch::Tensor T_to_lat,
    torch::Tensor G_inv,
    torch::Tensor G,
    torch::Tensor vlut_table
) {
    // Get dimensions
    int batch_size = A_encoded.size(0);
    int m = A_encoded.size(1);
    int k = A_encoded.size(2) * 8;  // k is 8 times the encoded dimension
    int n = B_encoded.size(2);
    
    // Create output tensor
    auto result = torch::zeros({batch_size, m, n}, torch::TensorOptions().dtype(torch::kFloat32).device(A_encoded.device()));
    
    // Configure grid and block dimensions
    dim3 blockDim(16, 16);
    dim3 gridDim((m + blockDim.x - 1) / blockDim.x, (n + blockDim.y - 1) / blockDim.y, batch_size);
    
    // Calculate shared memory size
    int shared_mem_size = 2 * blockDim.x * blockDim.y * sizeof(float);
    
    // Launch kernel
    cuda_e8_batched_vlut_twosided_matmul_optimized<<<gridDim, blockDim, shared_mem_size>>>(
        A_encoded.data_ptr<int>(),
        B_encoded.data_ptr<int>(),
        result.data_ptr<float>(),
        batch_size,
        m,
        k,
        n,
        q,
        M_levels,
        T_to_lat.data_ptr<float>(),
        G_inv.data_ptr<float>(),
        G.data_ptr<float>(),
        vlut_table.data_ptr<float>()
    );
    
    return result;
}

// PYBIND11_MODULE for Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cuda_e8_vector_dot_product_optimized", &cuda_e8_vector_dot_product_optimized_wrapper, "CUDA E8 optimized vector dot product");
    m.def("cuda_e8_matrix_vector_product_optimized", &cuda_e8_matrix_vector_product_optimized_wrapper, "CUDA E8 optimized matrix-vector product");
    m.def("cuda_e8_batched_matrix_multiply_optimized", &cuda_e8_batched_matrix_multiply_optimized_wrapper, "CUDA E8 optimized batched matrix multiplication");
    m.def("cuda_e8_batched_vlut_onesided_matmul_optimized", &cuda_e8_batched_vlut_onesided_matmul_optimized_wrapper, "CUDA E8 optimized batched vLUT one-sided matmul");
    m.def("cuda_e8_batched_vlut_twosided_matmul_optimized", &cuda_e8_batched_vlut_twosided_matmul_optimized_wrapper, "CUDA E8 optimized batched vLUT two-sided matmul");
}
