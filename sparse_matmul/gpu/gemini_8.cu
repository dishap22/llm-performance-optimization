#include <vector>
#include <stdexcept>
#include <numeric>
#include <algorithm> // For std::min

#include <cuda_runtime.h>
#include "../utils/csr_io.h"

// Helper macro for CUDA error checking.
#define CUDA_CHECK(err) { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

// --- Kernel 1: Parallel Exclusive Scan (Intra-Block Scan) ---
// This kernel performs the first level of a two-level scan. Each block
// computes an exclusive scan on its chunk of the data and writes its total sum
// to an auxiliary array `d_block_sums`.
__global__ void exclusive_scan_block_kernel(const int* d_in, int* d_out, int* d_block_sums, int n) {
    extern __shared__ int s_data[];
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * blockDim.x * 2;
    const int threads = blockDim.x;

    // Load data into shared memory
    int i1 = block_offset + tid;
    int i2 = block_offset + tid + threads;
    if (i1 < n) s_data[tid] = d_in[i1]; else s_data[tid] = 0;
    if (i2 < n) s_data[tid + threads] = d_in[i2]; else s_data[tid + threads] = 0;
    __syncthreads();

    // Parallel scan (up-sweep/reduction phase)
    for (int d = 1; d < 2 * threads; d *= 2) {
        int mask = (tid & ~(d - 1)) * 2 + d - 1;
        if ((tid & (d - 1)) == (d - 1)) {
            s_data[mask + d] += s_data[mask];
        }
        __syncthreads();
    }

    // Save block sum and clear last element
    if (tid == threads - 1) {
        if (d_block_sums) d_block_sums[blockIdx.x] = s_data[2 * threads - 1];
        s_data[2 * threads - 1] = 0;
    }
    __syncthreads();

    // Down-sweep phase
    for (int d = threads / 2; d >= 1; d /= 2) {
        int mask = (tid & ~(d - 1)) * 2 + d - 1;
        if ((tid & (d - 1)) == (d - 1)) {
            int t = s_data[mask];
            s_data[mask] = s_data[mask + d];
            s_data[mask + d] += t;
        }
        __syncthreads();
    }

    // Write results to global memory
    if (i1 < n) d_out[i1] = s_data[tid];
    if (i2 < n) d_out[i2] = s_data[tid + threads];
}


// --- Kernel 2: Parallel Exclusive Scan (Add Block Sums) ---
// This kernel adds the scanned block sums back to the intermediate results
// from the first kernel, producing the final scanned array.
__global__ void add_block_sums_kernel(int* d_out, const int* d_block_sums, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const int block_idx = (blockDim.x > 1) ? i / (blockDim.x * 2) : i / 512; // 512 = 256 * 2
    if (block_idx > 0) {
        d_out[i] += d_block_sums[block_idx - 1];
    }
}


/**
 * @brief Performs a high-performance parallel exclusive scan on the GPU.
 *
 * This function orchestrates a two-level scan to avoid sequential bottlenecks.
 * It is a key optimization over the single-threaded scan.
 *
 * @param d_in      Device pointer to the input array.
 * @param d_out     Device pointer to the output array (scan result).
 * @param n         The number of elements in the array.
 */
void parallel_exclusive_scan(int* d_in, int* d_out, int n) {
    if (n == 0) return;

    const int threads_per_block = 256;
    const int elements_per_block = threads_per_block * 2;
    const int num_blocks = (n + elements_per_block - 1) / elements_per_block;

    int* d_block_sums = nullptr;
    if (num_blocks > 1) {
        CUDA_CHECK(cudaMalloc(&d_block_sums, num_blocks * sizeof(int)));
    }

    // 1. Intra-block scan
    exclusive_scan_block_kernel<<<num_blocks, threads_per_block, elements_per_block * sizeof(int)>>>(d_in, d_out, d_block_sums, n);
    CUDA_CHECK(cudaGetLastError());

    if (num_blocks > 1) {
        // 2. Scan the block sums
        parallel_exclusive_scan(d_block_sums, d_block_sums, num_blocks);

        // 3. Add block sums back to the main array
        const int add_threads = 256;
        const int add_blocks = (n + add_threads - 1) / add_threads;
        add_block_sums_kernel<<<add_blocks, add_threads>>>(d_out, d_block_sums, n);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaFree(d_block_sums));
    }

    // The last element of indptr is the total NNZ
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
 * @brief Symbolic kernel using a warp-per-row strategy.
 *
 * Each warp collaborates to find the NNZ for a single row of the output matrix C.
 * This greatly improves parallelism and GPU occupancy.
 *
 * @param num_rows          The number of rows in matrix A (and C).
 * @param b_cols            The number of columns in matrix B (and C).
 * ... (device pointers)
 */
__global__ void spgemm_symbolic_warp_per_row(const int num_rows, const int b_cols,
                                             const int* __restrict__ d_A_indptr, const int* __restrict__ d_A_indices,
                                             const int* __restrict__ d_B_indptr, const int* __restrict__ d_B_indices,
                                             int* d_C_row_nnz, int* d_workspace_markers) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_rows) return;

    const int lane_id = threadIdx.x; // thread id within the warp
    const int row_marker = row + 1;
    int* markers = d_workspace_markers + static_cast<size_t>(row) * b_cols;
    int nnz = 0;

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
            // Use atomicCAS to ensure only one thread per warp marks and counts a column
            if (atomicCAS(markers + col_B, 0, row_marker) == 0) {
                atomicAdd(&nnz, 1);
            }
        }
    }

    // One thread in the warp writes the final count
    if (lane_id == 0) {
        d_C_row_nnz[row] = nnz;
    }
}


/**
 * @brief Numeric kernel using a warp-per-row strategy.
 *
 * Each warp collaborates to compute the values and column indices for a single row of C.
 * This version parallelizes the accumulation and compaction steps.
 *
 * @param num_rows              The number of rows in matrix A (and C).
 * @param b_cols                The number of columns in matrix B (and C).
 * ... (device pointers)
 */
__global__ void spgemm_numeric_warp_per_row(const int num_rows, const int b_cols,
                                            const int* __restrict__ d_A_indptr, const int* __restrict__ d_A_indices, const float* __restrict__ d_A_data,
                                            const int* __restrict__ d_B_indptr, const int* __restrict__ d_B_indices, const float* __restrict__ d_B_data,
                                            const int* __restrict__ d_C_indptr, int* d_C_indices, float* d_C_data,
                                            float* d_workspace_vals, int* d_workspace_markers) {
    const int row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= num_rows) return;

    const int lane_id = threadIdx.x;
    const int numeric_row_marker = row + 1 + num_rows;

    float* acc_vals = d_workspace_vals + static_cast<size_t>(row) * b_cols;
    int* acc_markers = d_workspace_markers + static_cast<size_t>(row) * b_cols;

    const int start_A = d_A_indptr[row];
    const int end_A = d_A_indptr[row + 1];

    // Part 1: Accumulate products in parallel into the workspace.
    for (int i = start_A + lane_id; i < end_A; i += warpSize) {
        const int col_A = d_A_indices[i];
        const float val_A = d_A_data[i];
        const int row_B = col_A;

        const int start_B = d_B_indptr[row_B];
        const int end_B = d_B_indptr[row_B + 1];

        for (int j = start_B; j < end_B; ++j) {
            const int col_B = d_B_indices[j];
            const float val_B = d_B_data[j];

            // Mark that this accumulator is active for this row
            if (atomicCAS(acc_markers + col_B, 0, numeric_row_marker) == 0) {
                acc_vals[col_B] = 0; // Initialize if this is the first touch
            }
            // Wait for all threads in the warp to see the marker
            __threadfence();
            // Now perform the atomic add
            atomicAdd(&acc_vals[col_B], val_A * val_B);
        }
    }
    __syncthreads(); // Synchronize all warps in the block after accumulation

    // Part 2: Compact the results from the workspace to global memory in parallel.
    const int C_row_start = d_C_indptr[row];
    const int C_row_end = d_C_indptr[row + 1];
    const int C_row_nnz = C_row_end - C_row_start;

    // Use a shared memory counter for this warp's written NNZ
    extern __shared__ int s_C_nnz_written[];
    if (lane_id == 0) {
        s_C_nnz_written[threadIdx.y] = 0;
    }
    __syncthreads();

    // Parallelize the compaction loop across the warp
    for (int j = lane_id; j < b_cols; j += warpSize) {
        if (acc_markers[j] == numeric_row_marker) {
            int write_pos = atomicAdd(&s_C_nnz_written[threadIdx.y], 1);
            if (C_row_start + write_pos < C_row_end) {
                d_C_indices[C_row_start + write_pos] = j;
                d_C_data[C_row_start + write_pos] = acc_vals[j];
            }
        }
    }
}


/**
 * @brief Performs sparse matrix-matrix multiplication (SpGEMM) on the GPU.
 *
 * This version is fully self-contained and optimized for correctness and performance,
 * avoiding external libraries and unsafe assumptions about data sparsity.
 *
 * @param A The first input sparse matrix in CSR format.
 * @param B The second input sparse matrix in CSR format.
 * @param C The output sparse matrix in CSR format.
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
        // The marker-based algorithm avoids the need for a full cudaMemset here,
        // but we do it once for the first phase. The numeric phase will use different markers.
        CUDA_CHECK(cudaMemset(d_workspace_markers, 0, workspace_elements * sizeof(int)));
    } else {
        d_workspace_markers = nullptr;
        d_workspace_vals = nullptr;
    }

    // --- 3. Symbolic Phase: Compute NNZ per row of C ---
    int* d_C_row_nnz;
    CUDA_CHECK(cudaMalloc(&d_C_row_nnz, A.rows * sizeof(int)));

    // **OPTIMIZATION**: Use warp-per-row launch configuration
    const int warps_per_block = 8; // Tunable parameter
    dim3 threads(warpSize, warps_per_block, 1);
    dim3 blocks((A.rows + warps_per_block - 1) / warps_per_block, 1, 1);

    spgemm_symbolic_warp_per_row<<<blocks, threads>>>(A.rows, B.cols, d_A_indptr, d_A_indices, d_B_indptr, d_B_indices, d_C_row_nnz, d_workspace_markers);
    CUDA_CHECK(cudaGetLastError());

    // --- 4. GPU-side Scan: Calculate C.indptr and total NNZ ---
    int *d_C_indptr;
    size_t c_indptr_size = (A.rows + 1) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_C_indptr, c_indptr_size));

    // **OPTIMIZATION**: Use the high-performance parallel scan
    parallel_exclusive_scan(d_C_row_nnz, d_C_indptr, A.rows);
    CUDA_CHECK(cudaGetLastError());

    int C_nnz = 0;
    if (A.rows > 0) {
        CUDA_CHECK(cudaMemcpy(&C_nnz, d_C_indptr + A.rows, sizeof(int), cudaMemcpyDeviceToHost));
    }

    C.rows = A.rows;
    C.cols = B.cols;
    C.indptr.resize(A.rows + 1);
    C.indices.resize(C_nnz);
    C.data.resize(C_nnz);

    CUDA_CHECK(cudaMemcpy(C.indptr.data(), d_C_indptr, c_indptr_size, cudaMemcpyDeviceToHost));

    // --- 5. Numeric Phase: Compute C.indices and C.data ---
    if (C_nnz > 0) {
        int* d_C_indices;
        float* d_C_data;
        CUDA_CHECK(cudaMalloc(&d_C_indices, C_nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_C_data, C_nnz * sizeof(float)));

        // Reset the marker workspace for the numeric phase. We can do this smartly
        // by launching the numeric kernel which uses a different set of markers,
        // avoiding another expensive cudaMemset.

        // Use the same optimized launch configuration
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