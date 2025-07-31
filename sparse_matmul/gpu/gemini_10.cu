#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm> // For std::min

#include <cuda_runtime.h>

#include "spgemm_interface.h"

// Helper macro for CUDA error checking.
#define CUDA_CHECK(err) { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

/**
 * @brief A robust, standard implementation of a block-level parallel exclusive scan.
 *
 * This kernel performs an exclusive scan on a segment of an array using shared memory.
 * It processes 2 elements per thread for higher efficiency. It also outputs the total
 * sum of the block's elements to an auxiliary array for use in a multi-block scan.
 *
 * @param d_in          Device pointer to the input array.
 * @param d_out         Device pointer to the output array (scan result).
 * @param d_block_sums  Device pointer to an array where the sum of this block's elements will be stored.
 * @param n             The number of elements in the input array `d_in`.
 */
__global__ void exclusive_scan_block_kernel(const int* d_in, int* d_out, int* d_block_sums, int n) {
    extern __shared__ int s_data[];
    const int tid = threadIdx.x;
    const int threads = blockDim.x;

    // Load two elements per thread into shared memory
    int i1 = blockIdx.x * (2 * threads) + tid;
    int i2 = i1 + threads;
    s_data[tid] = (i1 < n) ? d_in[i1] : 0;
    s_data[tid + threads] = (i2 < n) ? d_in[i2] : 0;
    __syncthreads();

    // Up-sweep (reduction) phase in shared memory
    for (unsigned int d = 1; d < 2 * threads; d *= 2) {
        __syncthreads();
        if (tid < threads) {
            int idx1 = (tid + 1) * 2 * d - 1;
            int idx2 = idx1 - d;
            if (idx1 < 2 * threads) {
                s_data[idx1] += s_data[idx2];
            }
        }
    }

    // Save block sum for the next phase and clear the last element for exclusive scan
    if (tid == 0) {
        if (d_block_sums != nullptr) {
            d_block_sums[blockIdx.x] = s_data[2 * threads - 1];
        }
        s_data[2 * threads - 1] = 0;
    }
    __syncthreads();

    // Down-sweep phase in shared memory
    for (unsigned int d = threads; d >= 1; d /= 2) {
        __syncthreads();
        if (tid < threads) {
            int idx1 = (tid + 1) * 2 * d - 1;
            int idx2 = idx1 - d;
            if (idx1 < 2 * threads) {
                int temp = s_data[idx2];
                s_data[idx2] = s_data[idx1];
                s_data[idx1] += temp;
            }
        }
    }
    __syncthreads();

    // Write results to global memory
    if (i1 < n) d_out[i1] = s_data[tid];
    if (i2 < n) d_out[i2] = s_data[tid + threads];
}


/**
 * @brief Adds the scanned block sums back to the intermediate results.
 *
 * This is the final step in a multi-block parallel scan.
 */
__global__ void add_block_sums_kernel(int* d_out, const int* d_scanned_block_sums, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int elements_per_block = 512; // Must match 2 * threads_per_block from the host
    const int block_idx = i / elements_per_block;
    if (block_idx > 0) {
        d_out[i] += d_scanned_block_sums[block_idx - 1];
    }
}


/**
 * @brief Performs a high-performance, non-recursive parallel exclusive scan on the GPU.
 *
 * This function orchestrates a multi-kernel scan to avoid sequential bottlenecks and
 * recursive calls that can confuse the NVCC compiler.
 */
void parallel_exclusive_scan(int* d_in, int* d_out, int n) {
    if (n == 0) {
        // Handle empty input: set the total NNZ to 0.
        int total_nnz = 0;
        CUDA_CHECK(cudaMemcpy(d_out, &total_nnz, sizeof(int), cudaMemcpyHostToDevice));
        return;
    }

    const int threads_per_block = 256;
    const int elements_per_block = threads_per_block * 2;
    size_t shared_mem_size = elements_per_block * sizeof(int);

    const int num_blocks = (n + elements_per_block - 1) / elements_per_block;

    // 1. First-level scan on the main data array.
    int* d_block_sums = nullptr;
    if (num_blocks > 1) {
        CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(int)));
    }
    exclusive_scan_block_kernel<<<num_blocks, threads_per_block, shared_mem_size>>>(d_in, d_out, d_block_sums, n);
    CUDA_CHECK(cudaGetLastError());

    if (num_blocks > 1) {
        // 2. Scan the block sums array.
        int* d_scanned_block_sums;
        CUDA_CHECK(cudaMalloc(&d_scanned_block_sums, num_blocks * sizeof(int)));

        const int num_sum_blocks = (num_blocks + elements_per_block - 1) / elements_per_block;
        // This scan does not need to output its own block sums (pass nullptr).
        exclusive_scan_block_kernel<<<num_sum_blocks, threads_per_block, shared_mem_size>>>(d_block_sums, d_scanned_block_sums, nullptr, num_blocks);
        CUDA_CHECK(cudaGetLastError());

        // 3. Add the scanned block sums back to the main array's intermediate results.
        const int add_threads = 256;
        const int add_blocks = (n + add_threads - 1) / add_threads;
        add_block_sums_kernel<<<add_blocks, add_threads>>>(d_out, d_scanned_block_sums, n);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_block_sums));
        CUDA_CHECK(cudaFree(d_scanned_block_sums));
    }

    // Final step: calculate total NNZ and write to the end of the output array.
    int total_nnz = 0;
    if (n > 0) {
        int last_val;
        CUDA_CHECK(cudaMemcpy(&last_val, d_in + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
        int last_scan_val;
        CUDA_CHECK(cudaMemcpy(&last_scan_val, d_out + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
        total_nnz = last_val + last_scan_val;
        CUDA_CHECK(cudaMemcpy(d_out + n, &total_nnz, sizeof(int), cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy(d_out + n, &total_nnz, sizeof(int), cudaMemcpyHostToDevice));
    }
}


/**
 * @brief Corrected symbolic kernel using a warp-per-row strategy.
 */
__global__ void spgemm_symbolic_warp_per_row(const int num_rows, const int b_cols,
                                             const int* __restrict__ d_A_indptr, const int* __restrict__ d_A_indices,
                                             const int* __restrict__ d_B_indptr, const int* __restrict__ d_B_indices,
                                             int* d_C_row_nnz, int* d_workspace_markers) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_rows) return;

    const int lane_id = threadIdx.x;
    int* row_markers = d_workspace_markers + static_cast<size_t>(row) * b_cols;
    int nnz = 0; // Each thread counts its successful marks

    const int start_A = d_A_indptr[row];
    const int end_A = d_A_indptr[row + 1];

    for (int i = start_A + lane_id; i < end_A; i += warpSize) {
        const int col_A = d_A_indices[i];
        const int row_B = col_A;
        const int start_B = d_B_indptr[row_B];
        const int end_B = d_B_indptr[row_B + 1];

        for (int j = start_B; j < end_B; ++j) {
            const int col_B = d_B_indices[j];
            if (atomicCAS(row_markers + col_B, 0, 1) == 0) {
                nnz++;
            }
        }
    }

    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        nnz += __shfl_down_sync(0xFFFFFFFF, nnz, offset);
    }

    if (lane_id == 0) {
        d_C_row_nnz[row] = nnz;
    }
}


/**
 * @brief Corrected numeric kernel using a warp-per-row strategy.
 */
__global__ void spgemm_numeric_warp_per_row(const int num_rows, const int b_cols,
                                            const int* __restrict__ d_A_indptr, const int* __restrict__ d_A_indices, const float* __restrict__ d_A_data,
                                            const int* __restrict__ d_B_indptr, const int* __restrict__ d_B_indices, const float* __restrict__ d_B_data,
                                            const int* __restrict__ d_C_indptr, int* d_C_indices, float* d_C_data,
                                            float* d_workspace_vals, int* d_workspace_markers) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_rows) return;

    const int lane_id = threadIdx.x;
    const int warp_id = threadIdx.y;

    float* row_vals = d_workspace_vals + static_cast<size_t>(row) * b_cols;
    int* row_markers = d_workspace_markers + static_cast<size_t>(row) * b_cols;
    const int symbolic_visited_flag = 1;

    // Part 1a: Initialize workspace values to 0 for this row's active columns
    for (int j = lane_id; j < b_cols; j += warpSize) {
        if (row_markers[j] == symbolic_visited_flag) {
            row_vals[j] = 0.0f;
        }
    }
    __syncthreads();

    // Part 1b: Accumulate products in parallel into the workspace.
    const int start_A = d_A_indptr[row];
    const int end_A = d_A_indptr[row + 1];

    for (int i = start_A + lane_id; i < end_A; i += warpSize) {
        const int col_A = d_A_indices[i];
        const float val_A = d_A_data[i];
        const int row_B = col_A;

        const int start_B = d_B_indptr[row_B];
        const int end_B = d_B_indptr[row_B + 1];

        for (int j = start_B; j < end_B; ++j) {
            const int col_B = d_B_indices[j];
            const float val_B = d_B_data[j];
            if (row_markers[col_B] == symbolic_visited_flag) {
                 atomicAdd(&row_vals[col_B], val_A * val_B);
            }
        }
    }
    __syncthreads();

    // Part 2: Compact the results from the workspace to global memory in parallel.
    const int C_row_start = d_C_indptr[row];
    const int C_row_end = d_C_indptr[row + 1];

    extern __shared__ int s_C_nnz_written[];
    if (lane_id == 0) {
        s_C_nnz_written[warp_id] = 0;
    }
    __syncthreads();

    for (int j = lane_id; j < b_cols; j += warpSize) {
        if (row_markers[j] == symbolic_visited_flag) {
            int write_pos = atomicAdd(&s_C_nnz_written[warp_id], 1);
            if (C_row_start + write_pos < C_row_end) {
                d_C_indices[C_row_start + write_pos] = j;
                d_C_data[C_row_start + write_pos] = row_vals[j];
            }
        }
    }
}


/**
 * @brief Main host function for GPU-based SpGEMM.
 */
void spgemm_gpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    if (A.cols != B.rows) {
        throw std::runtime_error("Incompatible matrix dimensions for SpGEMM (A.cols must equal B.rows).");
    }

    // --- 1. Device Memory Allocation & H2D Transfer for Inputs ---
    int *d_A_indptr, *d_A_indices, *d_B_indptr, *d_B_indices;
    float *d_A_data, *d_B_data;

    CUDA_CHECK(cudaMalloc(&d_A_indptr, (A.rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_A_indices, A.indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_A_data, A.data.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_indptr, (B.rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B_indices, B.indices.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_B_data, B.data.size() * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A_indptr, A.indptr.data(), (A.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_indices, A.indices.data(), A.indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_data, A.data.data(), A.data.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_indptr, B.indptr.data(), (B.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_indices, B.indices.data(), B.indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_data, B.data.data(), B.data.size() * sizeof(float), cudaMemcpyHostToDevice));

    // --- 2. Allocate Persistent Workspaces ---
    int* d_workspace_markers;
    float* d_workspace_vals;
    size_t workspace_elements = static_cast<size_t>(A.rows) * B.cols;
    if (workspace_elements > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace_markers, workspace_elements * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_workspace_vals, workspace_elements * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_workspace_markers, 0, workspace_elements * sizeof(int)));
    } else {
        d_workspace_markers = nullptr;
        d_workspace_vals = nullptr;
    }

    // --- 3. Symbolic Phase: Compute NNZ per row of C ---
    int* d_C_row_nnz;
    CUDA_CHECK(cudaMalloc(&d_C_row_nnz, A.rows * sizeof(int)));

    const int warps_per_block = 8; // Tunable parameter
    dim3 threads(warpSize, warps_per_block, 1);
    dim3 blocks((A.rows + warps_per_block - 1) / warps_per_block, 1, 1);

    spgemm_symbolic_warp_per_row<<<blocks, threads>>>(A.rows, B.cols, d_A_indptr, d_A_indices, d_B_indptr, d_B_indices, d_C_row_nnz, d_workspace_markers);
    CUDA_CHECK(cudaGetLastError());

    // --- 4. GPU-side Scan: Calculate C.indptr and total NNZ ---
    int *d_C_indptr;
    size_t c_indptr_size = (A.rows + 1) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_C_indptr, c_indptr_size));

    parallel_exclusive_scan(d_C_row_nnz, d_C_indptr, A.rows);
    CUDA_CHECK(cudaGetLastError());

    int C_nnz = 0;
    if (A.rows > 0) {
        CUDA_CHECK(cudaMemcpy(&C_nnz, d_C_indptr + A.rows, sizeof(int), cudaMemcpyDeviceToHost));
    }

    C.rows = A.rows;
    C.cols = B.cols;
    C.indptr.resize(A.rows + 1);
    if (C_nnz > 0) {
        C.indices.resize(C_nnz);
        C.data.resize(C_nnz);
    }

    CUDA_CHECK(cudaMemcpy(C.indptr.data(), d_C_indptr, c_indptr_size, cudaMemcpyDeviceToHost));

    // --- 5. Numeric Phase: Compute C.indices and C.data ---
    if (C_nnz > 0) {
        int* d_C_indices;
        float* d_C_data;
        CUDA_CHECK(cudaMalloc(&d_C_indices, C_nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_C_data, C_nnz * sizeof(float)));

        size_t shared_mem_size = warps_per_block * sizeof(int);
        spgemm_numeric_warp_per_row<<<blocks, threads, shared_mem_size>>>(A.rows, B.cols,
                                                               d_A_indptr, d_A_indices, d_A_data,
                                                               d_B_indptr, d_B_indices, d_B_data,
                                                               d_C_indptr, d_C_indices, d_C_data,
                                                               d_workspace_vals, d_workspace_markers);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(C.indices.data(), d_C_indices, C_nnz * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(C.data.data(), d_C_data, C_nnz * sizeof(float), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_C_indices));
        CUDA_CHECK(cudaFree(d_C_data));
    }

    // --- 6. Cleanup ---
    CUDA_CHECK(cudaFree(d_A_indptr));
    CUDA_CHECK(cudaFree(d_A_indices));
    CUDA_CHECK(cudaFree(d_A_data));
    CUDA_CHECK(cudaFree(d_B_indptr));
    CUDA_CHECK(cudaFree(d_B_indices));
    CUDA_CHECK(cudaFree(d_B_data));
    CUDA_CHECK(cudaFree(d_C_row_nnz));
    CUDA_CHECK(cudaFree(d_C_indptr));
    if (workspace_elements > 0) {
        CUDA_CHECK(cudaFree(d_workspace_markers));
        CUDA_CHECK(cudaFree(d_workspace_vals));
    }
}