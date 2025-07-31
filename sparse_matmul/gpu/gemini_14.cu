#include <vector>
#include <stdexcept>
#include <numeric>

#include <cuda_runtime.h>

// Use the correct relative path to the header file
// based on the project structure.
#include "../utils/csr_io.h"

// Helper macro for CUDA error checking.
#define CUDA_CHECK(err) { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

/**
 * @brief A simple, single-threaded kernel to perform an exclusive scan (prefix sum) on the GPU.
 *
 * This kernel replaces the need for external libraries, avoiding compiler compatibility issues.
 * It runs on a single thread but avoids expensive device-to-host data transfers.
 * NOTE: For very large matrices (e.g., millions of rows), this serial scan can become a
 * bottleneck. A parallel scan implementation would be a further optimization.
 *
 * @param d_in      Device pointer to the input array (e.g., row NNZ counts).
 * @param d_out     Device pointer to the output array (the result of the scan).
 * @param n         The number of elements in the array.
 */
__global__ void exclusive_scan_kernel(const int* d_in, int* d_out, int n) {
    // This kernel is intended to be launched with a single thread.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int sum = 0;
        for (int i = 0; i < n; i++) {
            int temp = d_in[i];
            d_out[i] = sum;
            sum += temp;
        }
        // The last element of indptr is the total NNZ
        d_out[n] = sum;
    }
}


/**
 * @brief Optimized symbolic kernel using a "block-per-row" strategy.
 *
 * Each thread block calculates the number of non-zero elements for a single row of C.
 * This approach significantly improves GPU utilization by launching one block per row,
 * ensuring that many SMs are active. Threads within the block parallelize the computation.
 *
 * @param num_rows          The number of rows in matrix A (and C).
 * @param b_cols            The number of columns in matrix B (and C).
 * @param d_A_indptr        Device pointer to the indptr array of matrix A.
 * @param d_A_indices       Device pointer to the indices array of matrix A.
 * @param d_B_indptr        Device pointer to the indptr array of matrix B.
 * @param d_B_indices       Device pointer to the indices array of matrix B.
 * @param d_C_row_nnz       Device pointer to an array to store the NNZ count for each row of C.
 * @param d_workspace_markers Device pointer to an integer workspace of size (num_rows * b_cols).
 */
__global__ void spgemm_symbolic_block_per_row(const int num_rows, const int b_cols,
                                              const int* __restrict__ d_A_indptr, const int* __restrict__ d_A_indices,
                                              const int* __restrict__ d_B_indptr, const int* __restrict__ d_B_indices,
                                              int* d_C_row_nnz, int* d_workspace_markers) {
    // Each block computes one row of C.
    const int row = blockIdx.x;
    if (row >= num_rows) return;

    const int row_marker = row + 1;
    // Each block works on its own slice of the global workspace.
    int* markers = d_workspace_markers + static_cast<size_t>(row) * b_cols;

    const int start_A = d_A_indptr[row];
    const int end_A = d_A_indptr[row + 1];

    // Parallelize the loop over A's non-zeros across threads in the block.
    for (int i = start_A + threadIdx.x; i < end_A; i += blockDim.x) {
        const int col_A = d_A_indices[i];
        const int row_B = col_A;
        const int start_B = d_B_indptr[row_B];
        const int end_B = d_B_indptr[row_B + 1];

        for (int j = start_B; j < end_B; ++j) {
            const int col_B = d_B_indices[j];
            // A non-atomic write is sufficient. If multiple threads write the same
            // marker, the outcome is identical (benign race condition).
            markers[col_B] = row_marker;
        }
    }

    // Synchronize all threads in the block to ensure all markers are written before counting.
    __syncthreads();

    // Count the marked elements in parallel.
    int thread_nnz = 0;
    for (int j = threadIdx.x; j < b_cols; j += blockDim.x) {
        if (markers[j] == row_marker) {
            thread_nnz++;
        }
    }

    // Reduce the counts from all threads in the block using shared memory.
    extern __shared__ int s_nnz[];
    s_nnz[threadIdx.x] = thread_nnz;
    __syncthreads();

    // Perform parallel reduction in shared memory.
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_nnz[threadIdx.x] += s_nnz[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final, total count for the row.
    if (threadIdx.x == 0) {
        d_C_row_nnz[row] = s_nnz[0];
    }
}


/**
 * @brief Corrected and optimized numeric kernel using a "block-per-row" strategy.
 *
 * This version fixes the race condition by using a zero-initialized workspace
 * and relying exclusively on atomicAdd for accumulation. This is thread-safe
 * and ensures numerical correctness.
 *
 * @param num_rows              The number of rows in matrix A (and C).
 * @param b_cols                The number of columns in matrix B (and C).
 * @param ...                   Device pointers for matrices A, B, and C.
 * @param d_workspace_vals      Device pointer to a float workspace, MUST be zero-initialized.
 * @param d_workspace_markers   Device pointer to an integer workspace from the symbolic phase.
 */
__global__ void spgemm_numeric_block_per_row(const int num_rows, const int b_cols,
                                             const int* __restrict__ d_A_indptr, const int* __restrict__ d_A_indices, const float* __restrict__ d_A_data,
                                             const int* __restrict__ d_B_indptr, const int* __restrict__ d_B_indices, const float* __restrict__ d_B_data,
                                             const int* __restrict__ d_C_indptr, int* d_C_indices, float* d_C_data,
                                             float* d_workspace_vals, int* d_workspace_markers) {
    // Each block computes one row of C.
    const int row = blockIdx.x;
    if (row >= num_rows) return;

    // The marker value set by the symbolic phase for this row.
    const int symbolic_marker = row + 1;

    // Each block works on its own slice of the global workspaces.
    float* acc_vals = d_workspace_vals + static_cast<size_t>(row) * b_cols;
    int* acc_markers = d_workspace_markers + static_cast<size_t>(row) * b_cols;

    const int start_A = d_A_indptr[row];
    const int end_A = d_A_indptr[row + 1];

    // Part 1: Accumulate products in parallel into the zero-initialized workspace.
    // This loop iterates over the non-zero elements of row A assigned to this thread.
    for (int i = start_A + threadIdx.x; i < end_A; i += blockDim.x) {
        const int col_A = d_A_indices[i];
        const float val_A = d_A_data[i];
        const int row_B = col_A;

        const int start_B = d_B_indptr[row_B];
        const int end_B = d_B_indptr[row_B + 1];

        // For each non-zero in A, iterate over the corresponding row in B.
        for (int j = start_B; j < end_B; ++j) {
            const int col_B = d_B_indices[j];
            const float val_B = d_B_data[j];

            // Atomically add the product to the accumulator for the corresponding output column.
            // Since the workspace is pre-initialized to zero, we can just use atomicAdd
            // for all updates, which is race-free.
            atomicAdd(&acc_vals[col_B], val_A * val_B);
        }
    }

    // Synchronize all threads in the block to ensure all atomic additions are globally visible
    // before the compaction phase begins.
    __syncthreads();

    // Part 2: Write the compacted results from the workspace to global memory.
    // This is done serially by a single thread (thread 0) in the block.
    if (threadIdx.x == 0) {
        const int C_row_start = d_C_indptr[row];
        int C_nnz_written = 0;
        // Scan through all possible columns for this row's slice of the workspace.
        for (int j = 0; j < b_cols; ++j) {
            // Use the marker from the symbolic phase to identify the non-zero columns.
            if (acc_markers[j] == symbolic_marker) {
                d_C_indices[C_row_start + C_nnz_written] = j;
                d_C_data[C_row_start + C_nnz_written] = acc_vals[j];
                C_nnz_written++;
            }
        }
    }
}


/**
 * @brief Performs sparse matrix-matrix multiplication (SpGEMM) on the GPU.
 *
 * This version is fully self-contained and optimized for performance by using a
 * "block-per-row" strategy, which ensures high GPU occupancy and numerical correctness.
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

    // --- 2. Allocate and Initialize Persistent Workspaces ---
    int* d_workspace_markers;
    float* d_workspace_vals;
    size_t workspace_elements = static_cast<size_t>(A.rows) * B.cols;
    if (workspace_elements > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace_markers, workspace_elements * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_workspace_vals, workspace_elements * sizeof(float)));
        // The marker-based algorithm relies on an initial zeroed state.
        CUDA_CHECK(cudaMemset(d_workspace_markers, 0, workspace_elements * sizeof(int)));
        // FIX: The value workspace must also be zero-initialized for the atomicAdd approach.
        CUDA_CHECK(cudaMemset(d_workspace_vals, 0, workspace_elements * sizeof(float)));
    } else {
        d_workspace_markers = nullptr;
        d_workspace_vals = nullptr;
    }

    // --- 3. Symbolic Phase: Compute NNZ per row of C ---
    int* d_C_row_nnz;
    CUDA_CHECK(cudaMalloc(&d_C_row_nnz, A.rows * sizeof(int)));

    // Set launch configuration for "block-per-row" strategy.
    int threads_per_block = 256;
    int blocks_per_grid = A.rows; // Launch one block for each row of A
    size_t shared_mem_size = threads_per_block * sizeof(int); // For reduction in symbolic kernel

    if (blocks_per_grid > 0) {
        spgemm_symbolic_block_per_row<<<blocks_per_grid, threads_per_block, shared_mem_size>>>(
            A.rows, B.cols, d_A_indptr, d_A_indices, d_B_indptr, d_B_indices, d_C_row_nnz, d_workspace_markers);
        CUDA_CHECK(cudaGetLastError());
    }

    // --- 4. GPU-side Scan: Calculate C.indptr and total NNZ ---
    int *d_C_indptr;
    size_t c_indptr_size = (A.rows + 1) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_C_indptr, c_indptr_size));

    if (A.rows > 0) {
        exclusive_scan_kernel<<<1, 1>>>(d_C_row_nnz, d_C_indptr, A.rows);
        CUDA_CHECK(cudaGetLastError());
    }

    int C_nnz = 0;
    if (A.rows > 0) {
        CUDA_CHECK(cudaMemcpy(&C_nnz, d_C_indptr + A.rows, sizeof(int), cudaMemcpyDeviceToHost));
    }

    C.rows = A.rows;
    C.cols = B.cols;
    C.indptr.resize(A.rows + 1);
    C.indices.resize(C_nnz);
    C.data.resize(C_nnz);

    if (A.rows > 0) {
        CUDA_CHECK(cudaMemcpy(C.indptr.data(), d_C_indptr, c_indptr_size, cudaMemcpyDeviceToHost));
    }

    // --- 5. Numeric Phase: Compute C.indices and C.data ---
    if (C_nnz > 0) {
        int* d_C_indices;
        float* d_C_data;
        CUDA_CHECK(cudaMalloc(&d_C_indices, C_nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_C_data, C_nnz * sizeof(float)));

        if (blocks_per_grid > 0) {
            spgemm_numeric_block_per_row<<<blocks_per_grid, threads_per_block>>>(
                A.rows, B.cols,
                d_A_indptr, d_A_indices, d_A_data,
                d_B_indptr, d_B_indices, d_B_data,
                d_C_indptr, d_C_indices, d_C_data,
                d_workspace_vals, d_workspace_markers);
            CUDA_CHECK(cudaGetLastError());
        }

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
