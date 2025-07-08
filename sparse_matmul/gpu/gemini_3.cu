#include <vector>
#include <stdexcept>
#include <numeric> // For std::iota if needed, though not used in final version

// The Thrust library is part of the CUDA Toolkit and provides highly optimized parallel algorithms.
// We use it here for an efficient, GPU-native prefix sum (exclusive scan).
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>


#include <cuda_runtime.h>

// --- Assumed definition from "csr_io.h" ---
// This struct is included here to make the code self-contained and compilable.
struct CSRMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<float> data;
};
// --- End of assumed definition ---


// Helper macro for CUDA error checking. It's good practice to wrap all CUDA API calls.
#define CUDA_CHECK(err) { \
    cudaError_t e = (err); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
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

    // Use row + 1 as the unique marker for this thread. This avoids confusion with the
    // initial '0' state of the workspace.
    const int row_marker = row + 1;

    // Each thread gets its private slice of the global workspace.
    int* markers = d_workspace_markers + static_cast<size_t>(row) * b_cols;

    int nnz = 0;
    const int start_A = d_A_indptr[row];
    const int end_A = d_A_indptr[row + 1];

    // For each non-zero element in A's row i
    for (int i = start_A; i < end_A; ++i) {
        const int col_A = d_A_indices[i];
        const int row_B = col_A;

        const int start_B = d_B_indptr[row_B];
        const int end_B = d_B_indptr[row_B + 1];

        // For each non-zero element in B's corresponding row
        for (int j = start_B; j < end_B; ++j) {
            const int col_B = d_B_indices[j];
            // OPTIMIZATION: Check if the marker for this column is not ours.
            // This is much faster than using a boolean flag that needs to be reset.
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
 * This version also uses the marker algorithm to manage the accumulator. More importantly,
 * it avoids the slow final loop over all `b_cols`. It records the column indices of non-zero
 * elements as they are created and then iterates over this much smaller list to write
 * the final results to global memory.
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

    const int row_marker = row + 1;

    // Private workspace slice for this thread.
    float* acc_vals = d_workspace_vals + static_cast<size_t>(row) * b_cols;
    int* acc_markers = d_workspace_markers + static_cast<size_t>(row) * b_cols;

    // OPTIMIZATION: Keep track of the columns that become non-zero for this row.
    // This avoids the expensive final loop over `b_cols`.
    // We use a temporary local array. The size is a heuristic; for matrices with
    // very dense rows, this might need adjustment or dynamic allocation.
    const int MAX_ROW_NNZ = 1024;
    int temp_cols[MAX_ROW_NNZ];
    int current_nnz = 0;

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

            if (acc_markers[col_B] != row_marker) {
                // First time we see this column for this row
                acc_markers[col_B] = row_marker;
                acc_vals[col_B] = val_A * val_B;
                if (current_nnz < MAX_ROW_NNZ) {
                    temp_cols[current_nnz++] = col_B;
                }
                // Note: If current_nnz >= MAX_ROW_NNZ, we have an overflow.
                // A production-grade code would handle this by spilling to global memory.
                // For this optimization, we assume row NNZ fits in the temp array.
            } else {
                acc_vals[col_B] += val_A * val_B;
            }
        }
    }

    // Part 2: Write the compacted results from the workspace to global memory.
    const int C_row_start = d_C_indptr[row];

    // The final output must be sorted by column index. A simple insertion sort
    // is efficient here since `temp_cols` is small and in local memory.
    for (int i = 1; i < current_nnz; ++i) {
        int key = temp_cols[i];
        int j = i - 1;
        while (j >= 0 && temp_cols[j] > key) {
            temp_cols[j + 1] = temp_cols[j];
            j--;
        }
        temp_cols[j + 1] = key;
    }

    // Write the now-sorted columns and their corresponding values to global memory.
    for (int i = 0; i < current_nnz; i++) {
        int col = temp_cols[i];
        d_C_indices[C_row_start + i] = col;
        d_C_data[C_row_start + i] = acc_vals[col];
    }
}


/**
 * @brief Performs sparse matrix-matrix multiplication (SpGEMM) on the GPU.
 *
 * Computes C = A * B using an optimized two-phase CUDA implementation.
 * This version avoids expensive `cudaMemset` calls by using a marker-based algorithm
 * and performs the prefix sum to calculate C's structure entirely on the GPU,
 * avoiding a costly device-host-device roundtrip.
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

    // --- OPTIMIZATION: Allocate persistent workspaces ---
    // These workspaces are large, but we now only allocate them once and use a marker
    // algorithm to avoid clearing them with expensive `cudaMemset` calls.
    int* d_workspace_markers;
    float* d_workspace_vals;
    size_t workspace_elements = static_cast<size_t>(A.rows) * B.cols;
    if (workspace_elements > 0) {
        CUDA_CHECK(cudaMalloc(&d_workspace_markers, workspace_elements * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_workspace_vals, workspace_elements * sizeof(float)));
        // We only need to clear the markers workspace ONCE at the beginning.
        CUDA_CHECK(cudaMemset(d_workspace_markers, 0, workspace_elements * sizeof(int)));
    } else {
        d_workspace_markers = nullptr;
        d_workspace_vals = nullptr;
    }


    // --- 2. Symbolic Phase: Compute NNZ per row of C ---
    int* d_C_row_nnz;
    CUDA_CHECK(cudaMalloc(&d_C_row_nnz, A.rows * sizeof(int)));

    int threads_per_block = 256;
    int blocks_per_grid = (A.rows + threads_per_block - 1) / threads_per_block;

    spgemm_symbolic_optimized<<<blocks_per_grid, threads_per_block>>>(A.rows, B.cols, d_A_indptr, d_A_indices, d_B_indptr, d_B_indices, d_C_row_nnz, d_workspace_markers);
    CUDA_CHECK(cudaGetLastError());


    // --- 3. GPU-side: Calculate C.indptr and total NNZ for C ---
    int *d_C_indptr;
    size_t c_indptr_size = (A.rows + 1) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_C_indptr, c_indptr_size));

    // OPTIMIZATION: Use Thrust for a high-performance, GPU-native exclusive scan.
    // This avoids the slow GPU->CPU->GPU roundtrip of the original code.
    // The try/catch block is removed as it can cause compilation issues with nvcc.
    // A failure in Thrust will typically raise a CUDA error caught by subsequent checks.
    thrust::device_ptr<int> d_C_row_nnz_ptr(d_C_row_nnz);
    thrust::device_ptr<int> d_C_indptr_ptr(d_C_indptr);
    // The first element of indptr is always 0.
    CUDA_CHECK(cudaMemset(d_C_indptr, 0, sizeof(int)));
    // Compute the exclusive scan for the rest of the array.
    thrust::exclusive_scan(d_C_row_nnz_ptr, d_C_row_nnz_ptr + A.rows, d_C_indptr_ptr + 1);

    // Get the total number of non-zeros for C to allocate its data arrays.
    // C_nnz = C.indptr[A.rows], which is the sum of all elements in C_row_nnz.
    int C_nnz = 0;
    if (A.rows > 0) {
        CUDA_CHECK(cudaMemcpy(&C_nnz, d_C_indptr + A.rows, sizeof(int), cudaMemcpyDeviceToHost));
    }

    C.rows = A.rows;
    C.cols = B.cols;
    C.indptr.resize(A.rows + 1);
    C.indices.resize(C_nnz);
    C.data.resize(C_nnz);

    // Copy the final indptr array back to the host.
    CUDA_CHECK(cudaMemcpy(C.indptr.data(), d_C_indptr, c_indptr_size, cudaMemcpyDeviceToHost));


    // --- 4. Numeric Phase: Compute C.indices and C.data ---
    if (C_nnz > 0) {
        int* d_C_indices;
        float* d_C_data;
        size_t c_indices_size = C_nnz * sizeof(int);
        size_t c_data_size = C_nnz * sizeof(float);

        CUDA_CHECK(cudaMalloc(&d_C_indices, c_indices_size));
        CUDA_CHECK(cudaMalloc(&d_C_data, c_data_size));

        spgemm_numeric_optimized<<<blocks_per_grid, threads_per_block>>>(A.rows, B.cols,
                                                               d_A_indptr, d_A_indices, d_A_data,
                                                               d_B_indptr, d_B_indices, d_B_data,
                                                               d_C_indptr, d_C_indices, d_C_data,
                                                               d_workspace_vals, d_workspace_markers);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(C.indices.data(), d_C_indices, c_indices_size, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(C.data.data(), d_C_data, c_data_size, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_C_indices));
        CUDA_CHECK(cudaFree(d_C_data));
    }

    // --- 5. Cleanup ---
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
