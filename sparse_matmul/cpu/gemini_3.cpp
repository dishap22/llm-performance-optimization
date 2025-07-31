#include "csr_io.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <omp.h>

// Optimized Single-Pass SpGEMM using a dense accumulator (Gustavson's Algorithm)
void spgemm_cpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    // Set the dimensions of the output matrix C
    C.rows = A.rows;
    C.cols = B.cols;
    C.indptr.resize(C.rows + 1, 0);

    // --- Single Pass with Intermediate Storage ---
    // We compute each row of C and store it temporarily. After the parallel computation,
    // we will assemble the final CSR matrix C from these temporary rows.
    std::vector<std::vector<std::pair<int, float>>> temp_rows(C.rows);

    #pragma omp parallel
    {
        // --- Per-Thread Workspace ---
        // Each thread gets its own dense accumulator and a mask to track non-zero entries.
        // The mask uses the current row index 'i' to mark used entries, avoiding a reset loop.
        // This is a common and efficient technique (the "iv-mask" trick).
        std::vector<float> accumulator(C.cols, 0.0f);
        std::vector<int> mask(C.cols, -1);
        std::vector<int> nnz_indices;
        nnz_indices.reserve(C.cols); // Pre-allocate memory to avoid reallocations

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < A.rows; ++i) {
            nnz_indices.clear(); // Reset the list of non-zero indices for the new row

            // Iterate over the non-zero elements of row 'i' in matrix A
            for (int jj = A.indptr[i]; jj < A.indptr[i + 1]; ++jj) {
                int j = A.indices[jj];      // Column index in A (which is the row index in B)
                float val_A = A.data[jj];

                // Iterate over the non-zero elements of row 'j' in matrix B
                for (int kk = B.indptr[j]; kk < B.indptr[j + 1]; ++kk) {
                    int k = B.indices[kk];      // Column index in B
                    float val_B = B.data[kk];

                    // If this is the first time we see column 'k' for this row 'i'
                    if (mask[k] != i) {
                        mask[k] = i;              // Mark it as visited for row 'i'
                        accumulator[k] = 0.0f;    // Ensure the accumulator is clean
                        nnz_indices.push_back(k); // Add to the list of non-zero columns for this row
                    }
                    accumulator[k] += val_A * val_B;
                }
            }

            // --- Store the computed row temporarily ---
            // The CSR format requires column indices to be sorted.
            std::sort(nnz_indices.begin(), nnz_indices.end());

            temp_rows[i].reserve(nnz_indices.size());
            for (int col_idx : nnz_indices) {
                temp_rows[i].push_back({col_idx, accumulator[col_idx]});
            }
        }
    }

    // --- Final Assembly of C (Serial) ---
    // Now that all rows have been computed in parallel, we assemble the final
    // CSR matrix from the temporary storage.

    // 1. Calculate the prefix sum to determine C.indptr
    int total_non_zeros = 0;
    for (int i = 0; i < C.rows; ++i) {
        C.indptr[i] = total_non_zeros;
        total_non_zeros += temp_rows[i].size();
    }
    C.indptr[C.rows] = total_non_zeros;

    // 2. Allocate final storage for C's indices and data
    C.indices.resize(total_non_zeros);
    C.data.resize(total_non_zeros);

    // 3. Copy the data from temporary storage to the final CSR arrays
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < C.rows; ++i) {
        int offset = C.indptr[i];
        for (size_t j = 0; j < temp_rows[i].size(); ++j) {
            C.indices[offset + j] = temp_rows[i][j].first;
            C.data[offset + j] = temp_rows[i][j].second;
        }
    }
}