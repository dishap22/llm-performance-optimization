#include <vector>
#include <stdexcept>
#include <numeric>

#include <cuda_runtime.h>

// CORRECT FIX: Use the correct relative path to the header file
// based on the project structure revealed by the compile command.
#include "../utils/csr_io.h"

// Helper macro for CUDA error checking. It's good practice to wrap all CUDA API calls.
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
 * It runs on a single thread but avoids expensive device-to-host data transfers, making it
 * faster than a CPU-based scan for calculating the indptr array.
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
 * @brief Optimized symbolic kernel to calculate the number of non-zero elements per row of C.
 *
 * This version uses a "marker" algorithm to avoid expensive cudaMemset operations on the workspace.
 * Each thread, responsible for a row, uses its unique row index + 1 as a marker.
 * It writes this marker to the workspace to signify that a column has been visited *for this specific row*.
 *
 * @param num_rows          The number of rows in matrix A (and C).
 * @param b_cols            The number of columns in matrix B (and C).
 * @param d_A_indptr        Device pointer to the indptr array of matrix A.
 * @param d_A_indices       Device pointer to the indices array of matrix A.
 * @param d_B_indptr        Device pointer to the indptr array of matrix B.
 * @param d_B_indices       Device pointer to the indices array of matrix B.
 * @param d_C_row_nnz       Device pointer to an array to store the NNZ count for each row of C.
 * @param d_workspace_markers Device pointer to an integer workspace of size (num_rows * b_cols),
 * pre-initialized to zero.
 */
__global__ void spgemm_symbolic_optimized(const int num_rows, const int b_cols,
                                          const int* __restrict__ d_A_indptr, const int* __restrict__ d_A_indices,
                                          const int* __restrict__ d_B_indptr, const int* __restrict__ d_B_indices,
                                          int* d_C_row_nnz, int* d_workspace_markers) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    const int row_marker = row + 1;
    int* markers = d_workspace_markers + static_cast<size_t>(row) * b_cols;
    int nnz = 0;
    const int start_A = d_A_indptr[row];
    const int end_A = d_A_indptr[row + 1];

    for (int i = start_A; i < end_A; ++i) {
        const int col_A = d_A_indices[i];
        const int row_B = col_A;
        const int start_B = d_B_indptr[row_B];
        const int end_B = d_B_indptr[row_B + 1];

        for (int j = start_B; j < end_B; ++j) {
            const int col_B = d_B_indices[j];
            if (markers[col_B] != row_marker) {
                markers[col_B] = row_marker;
                nnz++;
            }
        }
    }
    d_C_row_nnz[row] = nnz;
}


/**
 * @brief Optimized numeric kernel to compute the values and column indices of the output matrix C.
 *
 * This version uses the marker algorithm and records newly created non-zero column indices
 * in a small temporary array in local memory. This avoids a slow final loop over all possible
 * columns (`b_cols`) and makes the compaction step much more efficient.
 *
 * @param num_rows              The number of rows in matrix A (and C).
 * @param b_cols                The number of columns in matrix B (and C).
 * @param ...                   Device pointers for matrices A, B, and C.
 * @param d_workspace_vals      Device pointer to a float workspace for accumulating values.
 * @param d_workspace_markers   Device pointer to an integer workspace for tracking active accumulators.
 */
__global__ void spgemm_numeric_optimized(const int num_rows, const int b_cols,
                                         const int* __restrict__ d_A_indptr, const int* __restrict__ d_A_indices, const float* __restrict__ d_A_data,
                                         const int* __restrict__ d_B_indptr, const int* __restrict__ d_B_indices, const float* __restrict__ d_B_data,
                                         const int* __restrict__ d_C_indptr, int* d_C_indices, float* d_C_data,
                                         float* d_workspace_vals, int* d_workspace_markers) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    // BUG FIX: Use a different range of markers for the numeric phase to distinguish
    // it from the symbolic phase. The symbolic phase uses markers in [1, num_rows].
    // This phase will use markers in [num_rows + 1, 2 * num_rows].
    const int numeric_row_marker = row + 1 + num_rows;

    float* acc_vals = d_workspace_vals + static_cast<size_t>(row) * b_cols;
    int* acc_markers = d_workspace_markers + static_cast<size_t>(row) * b_cols;

    const int MAX_ROW_NNZ = 1024; // A heuristic limit for on-chip sorting.
    int temp_cols[MAX_ROW_NNZ];
    int current_nnz = 0;

    const int start_A = d_A_indptr[row];
    const int end_A = d_A_indptr[row + 1];

    for (int i = start_A; i < end_A; ++i) {
        const int col_A = d_A_indices[i];
        const float val_A = d_A_data[i];
        const int row_B = col_A;
        const int start_B = d_B_indptr[row_B];
        const int end_B = d_B_indptr[row_B + 1];

        for (int j = start_B; j < end_B; ++j) {
            const int col_B = d_B_indices[j];
            const float val_B = d_B_data[j];

            // This check now works correctly. The first time a column is seen for this
            // row in the numeric phase, the condition is true.
            if (acc_markers[col_B] != numeric_row_marker) {
                acc_markers[col_B] = numeric_row_marker;
                acc_vals[col_B] = val_A * val_B; // Correctly initializes the value.
                if (current_nnz < MAX_ROW_NNZ) {
                    temp_cols[current_nnz++] = col_B; // Correctly adds to compaction list.
                }
            } else {
                acc_vals[col_B] += val_A * val_B; // Correctly accumulates subsequent values.
            }
        }
    }

    const int C_row_start = d_C_indptr[row];

    // Sort the collected column indices locally before writing to global memory.
    for (int i = 1; i < current_nnz; ++i) {
        int key = temp_cols[i];
        int k = i - 1;
        while (k >= 0 && temp_cols[k] > key) {
            temp_cols[k + 1] = temp_cols[k];
            k--;
        }
        temp_cols[k + 1] = key;
    }

    // Write the sorted columns and their corresponding values to global memory.
    for (int i = 0; i < current_nnz; i++) {
        int col = temp_cols[i];
        d_C_indices[C_row_start + i] = col;
        d_C_data[C_row_start + i] = acc_vals[col];
    }
}


/**
 * @brief Performs sparse matrix-matrix multiplication (SpGEMM) on the GPU.
 *
 * This version is fully self-contained, avoiding external libraries like Thrust to
 * prevent compiler compatibility issues. It uses a custom kernel for the prefix-sum
 * calculation, keeping the entire process on the GPU.
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
        CUDA_CHECK(cudaMemset(d_workspace_markers, 0, workspace_elements * sizeof(int)));
    } else {
        d_workspace_markers = nullptr;
        d_workspace_vals = nullptr;
    }

    // --- 3. Symbolic Phase: Compute NNZ per row of C ---
    int* d_C_row_nnz;
    CUDA_CHECK(cudaMalloc(&d_C_row_nnz, A.rows * sizeof(int)));

    int threads_per_block = 256;
    int blocks_per_grid = (A.rows + threads_per_block - 1) / threads_per_block;

    spgemm_symbolic_optimized<<<blocks_per_grid, threads_per_block>>>(A.rows, B.cols, d_A_indptr, d_A_indices, d_B_indptr, d_B_indices, d_C_row_nnz, d_workspace_markers);
    CUDA_CHECK(cudaGetLastError());

    // --- 4. GPU-side Scan: Calculate C.indptr and total NNZ ---
    int *d_C_indptr;
    size_t c_indptr_size = (A.rows + 1) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_C_indptr, c_indptr_size));

    // Call the custom kernel to perform exclusive scan on the GPU.
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

    CUDA_CHECK(cudaMemcpy(C.indptr.data(), d_C_indptr, c_indptr_size, cudaMemcpyDeviceToHost));

    // --- 5. Numeric Phase: Compute C.indices and C.data ---
    if (C_nnz > 0) {
        int* d_C_indices;
        float* d_C_data;
        CUDA_CHECK(cudaMalloc(&d_C_indices, C_nnz * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_C_data, C_nnz * sizeof(float)));

        spgemm_numeric_optimized<<<blocks_per_grid, threads_per_block>>>(A.rows, B.cols,
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
