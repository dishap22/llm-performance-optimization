#include <vector>
#include <stdexcept>
#include "csr_io.h" // Contains the definition for CSRMatrix
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(err) { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

/**
 * @brief Symbolic kernel to calculate the number of non-zero elements per row of the output matrix C.
 *
 * Each thread computes the NNZ for one row of C = A * B. It uses a boolean workspace
 * as a hash map to mark visited column indices of B to count the unique resulting columns.
 *
 * @param num_rows          The number of rows in matrix A (and C).
 * @param b_cols            The number of columns in matrix B (and C).
 * @param d_A_indptr        Device pointer to the indptr array of matrix A.
 * @param d_A_indices       Device pointer to the indices array of matrix A.
 * @param d_B_indptr        Device pointer to the indptr array of matrix B.
 * @param d_B_indices       Device pointer to the indices array of matrix B.
 * @param d_C_row_nnz       Device pointer to an array to store the NNZ count for each row of C.
 * @param d_workspace       Device pointer to a boolean workspace of size (num_rows * b_cols).
 */
__global__ void spgemm_symbolic(const int num_rows, const int b_cols,
                                const int* __restrict__ d_A_indptr, const int* __restrict__ d_A_indices,
                                const int* __restrict__ d_B_indptr, const int* __restrict__ d_B_indices,
                                int* d_C_row_nnz, bool* d_workspace) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    // Each thread gets a private slice of the workspace to use as a boolean hash map.
    bool* visited = d_workspace + static_cast<size_t>(row) * b_cols;

    int nnz = 0;
    const int start_A = d_A_indptr[row];
    const int end_A = d_A_indptr[row + 1];

    // For each non-zero element in A's row i
    for (int i = start_A; i < end_A; ++i) {
        const int col_A = d_A_indices[i];
        const int row_B = col_A; // This becomes the row to inspect in B

        const int start_B = d_B_indptr[row_B];
        const int end_B = d_B_indptr[row_B + 1];

        // For each non-zero element in B's corresponding row
        for (int j = start_B; j < end_B; ++j) {
            const int col_B = d_B_indices[j];
            if (!visited[col_B]) {
                visited[col_B] = true;
                nnz++;
            }
        }
    }
    d_C_row_nnz[row] = nnz;
}

/**
 * @brief Numeric kernel to compute the values and column indices of the output matrix C.
 *
 * Each thread computes one row of C. It uses a float workspace as an accumulator for the values
 * and a boolean workspace to track which entries have been accumulated. After accumulation,
 * it writes the results to the final C matrix arrays.
 *
 * @param num_rows              The number of rows in matrix A (and C).
 * @param b_cols                The number of columns in matrix B (and C).
 * @param d_A_indptr, ..._data  Device pointers for matrix A.
 * @param d_B_indptr, ..._data  Device pointers for matrix B.
 * @param d_C_indptr, ..._data  Device pointers for the output matrix C.
 * @param d_workspace_vals      Device pointer to a float workspace for accumulating values.
 * @param d_workspace_flags     Device pointer to a boolean workspace for tracking active accumulators.
 */
__global__ void spgemm_numeric(const int num_rows, const int b_cols,
                               const int* __restrict__ d_A_indptr, const int* __restrict__ d_A_indices, const float* __restrict__ d_A_data,
                               const int* __restrict__ d_B_indptr, const int* __restrict__ d_B_indices, const float* __restrict__ d_B_data,
                               const int* __restrict__ d_C_indptr, int* d_C_indices, float* d_C_data,
                               float* d_workspace_vals, bool* d_workspace_flags) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    // Private workspace slice for this thread.
    float* acc_vals = d_workspace_vals + static_cast<size_t>(row) * b_cols;
    bool* acc_flags = d_workspace_flags + static_cast<size_t>(row) * b_cols;

    const int start_A = d_A_indptr[row];
    const int end_A = d_A_indptr[row + 1];

    // Part 1: Accumulate products for the current row.
    for (int i = start_A; i < end_A; ++i) {
        const int col_A = d_A_indices[i];
        const float val_A = d_A_data[i];
        const int row_B = col_A;

        const int start_B = d_B_indptr[row_B];
        const int end_B = d_B_indptr[row_B + 1];

        for (int j = start_B; j < end_B; ++j) {
            const int col_B = d_B_indices[j];
            const float val_B = d_B_data[j];

            if (!acc_flags[col_B]) {
                acc_flags[col_B] = true;
                acc_vals[col_B] = val_A * val_B;
            } else {
                acc_vals[col_B] += val_A * val_B;
            }
        }
    }

    // Part 2: Write the compacted results from the workspace to global memory.
    const int C_row_start = d_C_indptr[row];
    int C_nnz_written = 0;
    for (int j = 0; j < b_cols; ++j) {
        if (acc_flags[j]) {
            d_C_indices[C_row_start + C_nnz_written] = j;
            d_C_data[C_row_start + C_nnz_written] = acc_vals[j];
            C_nnz_written++;
        }
    }
}


/**
 * @brief Performs sparse matrix-matrix multiplication (SpGEMM) on the GPU.
 *
 * Computes C = A * B using a custom two-phase CUDA implementation.
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

    size_t a_indptr_size = (A.rows + 1) * sizeof(int);
    size_t a_indices_size = A.indices.size() * sizeof(int);
    size_t a_data_size = A.data.size() * sizeof(float);
    size_t b_indptr_size = (B.rows + 1) * sizeof(int);
    size_t b_indices_size = B.indices.size() * sizeof(int);
    size_t b_data_size = B.data.size() * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_A_indptr, a_indptr_size));
    CUDA_CHECK(cudaMalloc(&d_A_indices, a_indices_size));
    CUDA_CHECK(cudaMalloc(&d_A_data, a_data_size));
    CUDA_CHECK(cudaMalloc(&d_B_indptr, b_indptr_size));
    CUDA_CHECK(cudaMalloc(&d_B_indices, b_indices_size));
    CUDA_CHECK(cudaMalloc(&d_B_data, b_data_size));

    CUDA_CHECK(cudaMemcpy(d_A_indptr, A.indptr.data(), a_indptr_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_indices, A.indices.data(), a_indices_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_A_data, A.data.data(), a_data_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_indptr, B.indptr.data(), b_indptr_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_indices, B.indices.data(), b_indices_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_data, B.data.data(), b_data_size, cudaMemcpyHostToDevice));

    // --- 2. Symbolic Phase: Compute NNZ per row of C ---
    int* d_C_row_nnz;
    CUDA_CHECK(cudaMalloc(&d_C_row_nnz, A.rows * sizeof(int)));

    bool* d_symbolic_workspace;
    size_t workspace_size = static_cast<size_t>(A.rows) * B.cols;
    CUDA_CHECK(cudaMalloc(&d_symbolic_workspace, workspace_size * sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_symbolic_workspace, 0, workspace_size * sizeof(bool)));

    int threads_per_block = 256;
    int blocks_per_grid = (A.rows + threads_per_block - 1) / threads_per_block;

    spgemm_symbolic<<<blocks_per_grid, threads_per_block>>>(A.rows, B.cols, d_A_indptr, d_A_indices, d_B_indptr, d_B_indices, d_C_row_nnz, d_symbolic_workspace);
    CUDA_CHECK(cudaGetLastError());

    std::vector<int> h_C_row_nnz(A.rows);
    CUDA_CHECK(cudaMemcpy(h_C_row_nnz.data(), d_C_row_nnz, A.rows * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_symbolic_workspace));
    CUDA_CHECK(cudaFree(d_C_row_nnz));

    // --- 3. CPU-side: Calculate C.indptr and total NNZ for C ---
    C.indptr.resize(A.rows + 1, 0);
    int C_nnz = 0;
    for (int i = 0; i < A.rows; ++i) {
        C.indptr[i+1] = C.indptr[i] + h_C_row_nnz[i];
    }
    C_nnz = C.indptr[A.rows];
    C.rows = A.rows;
    C.cols = B.cols;
    C.indices.resize(C_nnz);
    C.data.resize(C_nnz);

    // --- 4. Numeric Phase: Compute C.indices and C.data ---
    if (C_nnz > 0) {
        int *d_C_indptr, *d_C_indices;
        float* d_C_data;
        float* d_numeric_workspace_vals;
        bool* d_numeric_workspace_flags;

        size_t c_indptr_size = (A.rows + 1) * sizeof(int);
        size_t c_indices_size = C_nnz * sizeof(int);
        size_t c_data_size = C_nnz * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_C_indptr, c_indptr_size));
        CUDA_CHECK(cudaMalloc(&d_C_indices, c_indices_size));
        CUDA_CHECK(cudaMalloc(&d_C_data, c_data_size));
        CUDA_CHECK(cudaMalloc(&d_numeric_workspace_vals, workspace_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_numeric_workspace_flags, workspace_size * sizeof(bool)));

        CUDA_CHECK(cudaMemcpy(d_C_indptr, C.indptr.data(), c_indptr_size, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_numeric_workspace_vals, 0, workspace_size * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_numeric_workspace_flags, 0, workspace_size * sizeof(bool)));

        spgemm_numeric<<<blocks_per_grid, threads_per_block>>>(A.rows, B.cols,
                                                               d_A_indptr, d_A_indices, d_A_data,
                                                               d_B_indptr, d_B_indices, d_B_data,
                                                               d_C_indptr, d_C_indices, d_C_data,
                                                               d_numeric_workspace_vals, d_numeric_workspace_flags);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(C.indices.data(), d_C_indices, c_indices_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(C.data.data(), d_C_data, c_data_size, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_C_indptr));
        CUDA_CHECK(cudaFree(d_C_indices));
        CUDA_CHECK(cudaFree(d_C_data));
        CUDA_CHECK(cudaFree(d_numeric_workspace_vals));
        CUDA_CHECK(cudaFree(d_numeric_workspace_flags));
    }

    // --- 5. Cleanup ---
    CUDA_CHECK(cudaFree(d_A_indptr));
    CUDA_CHECK(cudaFree(d_A_indices));
    CUDA_CHECK(cudaFree(d_A_data));
    CUDA_CHECK(cudaFree(d_B_indptr));
    CUDA_CHECK(cudaFree(d_B_indices));
    CUDA_CHECK(cudaFree(d_B_data));
}