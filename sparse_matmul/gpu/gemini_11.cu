#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm> // For std::min

#include <cuda_runtime.h>
#include <cub/cub.cuh> // **FIX**: Include the CUB library for parallel primitives.

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
 * @brief Corrected symbolic kernel using a warp-per-row strategy.
 *
 * Each warp collaborates to find the NNZ for a single row of C.
 * This version uses an efficient warp reduction and correct atomic operations.
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

    // Parallelize the outer loop over A's non-zeros across the warp
    for (int i = start_A + lane_id; i < end_A; i += warpSize) {
        const int col_A = d_A_indices[i];
        const int row_B = col_A;
        const int start_B = d_B_indptr[row_B];
        const int end_B = d_B_indptr[row_B + 1];

        for (int j = start_B; j < end_B; ++j) {
            const int col_B = d_B_indices[j];
            // Atomically mark the column. If we are the first thread to do so, count it.
            // This relies on the workspace slice being 0-initialized.
            if (atomicCAS(row_markers + col_B, 0, 1) == 0) {
                nnz++;
            }
        }
    }

    // Reduce the count across the warp. All threads in a warp add their nnz counts.
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        nnz += __shfl_down_sync(0xFFFFFFFF, nnz, offset);
    }

    // Lane 0 of the warp writes the final, aggregated count for the row.
    if (lane_id == 0) {
        d_C_row_nnz[row] = nnz;
    }
}


/**
 * @brief Corrected numeric kernel using a warp-per-row strategy.
 *
 * This version adds a parallel initialization step for the workspace and
 * uses correct atomic operations for accumulation and compaction.
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
    // The warp parallelizes this initialization step.
    for (int j = lane_id; j < b_cols; j += warpSize) {
        if (row_markers[j] == symbolic_visited_flag) {
            row_vals[j] = 0.0f;
        }
    }
    __syncthreads(); // Sync all warps in the block to ensure initialization is complete.

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
            // Check if this is a valid output column before accumulating
            if (row_markers[col_B] == symbolic_visited_flag) {
                 atomicAdd(&row_vals[col_B], val_A * val_B);
            }
        }
    }
    __syncthreads(); // Sync all warps in the block after accumulation

    // Part 2: Compact the results from the workspace to global memory in parallel.
    const int C_row_start = d_C_indptr[row];
    const int C_row_end = d_C_indptr[row + 1];

    // Use a shared memory counter for this warp's written NNZ
    extern __shared__ int s_C_nnz_written[];
    if (lane_id == 0) {
        s_C_nnz_written[warp_id] = 0;
    }
    __syncthreads();

    // Parallelize the compaction loop across the warp
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
 * This version uses CUB for the parallel scan, which is robust and highly optimized.
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

    // --- 4. GPU-side Scan: Calculate C.indptr and total NNZ using CUB ---
    int *d_C_indptr;
    size_t c_indptr_size = (A.rows + 1) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_C_indptr, c_indptr_size));

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    // First, get the size of the temporary storage CUB needs
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_C_row_nnz, d_C_indptr, A.rows);
    CUDA_CHECK(cudaGetLastError());
    // Allocate the temporary storage
    CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Now, perform the scan
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_C_row_nnz, d_C_indptr, A.rows);
    CUDA_CHECK(cudaGetLastError());

    int C_nnz = 0;
    if (A.rows > 0) {
        // To get the total NNZ, we need the last element of the indptr array plus the last element of the input array.
        int last_nnz_count;
        CUDA_CHECK(cudaMemcpy(&last_nnz_count, d_C_row_nnz + A.rows - 1, sizeof(int), cudaMemcpyDeviceToHost));
        int last_indptr_val;
        CUDA_CHECK(cudaMemcpy(&last_indptr_val, d_C_indptr + A.rows - 1, sizeof(int), cudaMemcpyDeviceToHost));
        C_nnz = last_indptr_val + last_nnz_count;
        // Copy the total NNZ to the last position of d_C_indptr on the device
        CUDA_CHECK(cudaMemcpy(d_C_indptr + A.rows, &C_nnz, sizeof(int), cudaMemcpyHostToDevice));
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
    CUDA_CHECK(cudaFree(d_temp_storage));
    if (workspace_elements > 0) {
        CUDA_CHECK(cudaFree(d_workspace_markers));
        CUDA_CHECK(cudaFree(d_workspace_vals));
    }
}
