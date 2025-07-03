#include "csr_io.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <omp.h>

// spgemm_cpu.cpp
void spgemm_cpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    // Set the dimensions of the output matrix C
    C.rows = A.rows;
    C.cols = B.cols;
    C.indptr.resize(C.rows + 1, 0);

    // This vector will store vectors of intermediate results for each row of C.
    // Each inner vector will hold the column indices and values for the non-zero elements of a single row.
    std::vector<std::vector<std::pair<int, float>>> intermediate_rows(C.rows);

    // The core of the SpGEMM computation is parallelized across the rows of matrix A.
    // Each thread computes a subset of the rows of the resulting matrix C.
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < A.rows; ++i) {
        // A temporary dense vector (or accumulator) for the current row i of C.
        // The `next` vector is used to keep track of occupied indices in the accumulator
        // to avoid iterating over all B.cols entries, which would be inefficient.
        std::vector<float> accumulator(B.cols, 0.0f);
        std::vector<int> next;
        next.reserve(A.indptr[i+1] - A.indptr[i]); // Pre-allocate memory as an optimization

        // Iterate over the non-zero elements of row i in matrix A
        for (int jj = A.indptr[i]; jj < A.indptr[i + 1]; ++jj) {
            int j = A.indices[jj];      // Column index in A (which is the row index in B)
            float val_A = A.data[jj];   // Value of A(i, j)

            // Now, iterate over the non-zero elements of row j in matrix B
            for (int kk = B.indptr[j]; kk < B.indptr[j + 1]; ++kk) {
                int k = B.indices[kk];      // Column index in B
                float val_B = B.data[kk];   // Value of B(j, k)

                // If this is the first time we've encountered this column index k for the current output row i,
                // add it to our list of non-zero candidates (`next`).
                if (accumulator[k] == 0.0f) {
                    next.push_back(k);
                }

                // Perform the scalar multiplication and add to the accumulator
                accumulator[k] += val_A * val_B;
            }
        }

        // Sort the column indices to ensure the CSR format is correct.
        std::sort(next.begin(), next.end());

        // Reserve space in the intermediate storage for this row to avoid reallocations.
        intermediate_rows[i].reserve(next.size());
        for (int col_idx : next) {
            // Only store actual non-zero values.
            if (accumulator[col_idx] != 0.0f) {
                intermediate_rows[i].push_back({col_idx, accumulator[col_idx]});
            }
        }
    }

    // At this point, all parallel computation is done. The following part constructs the final CSR matrix C
    // from the intermediate results. This part is serial.

    // Calculate the row pointers (indptr) for C by accumulating the number of non-zero elements in each row.
    for (int i = 0; i < C.rows; ++i) {
        C.indptr[i + 1] = C.indptr[i] + intermediate_rows[i].size();
    }

    // Get the total number of non-zero elements in C.
    int total_non_zeros = C.indptr[C.rows];
    C.indices.resize(total_non_zeros);
    C.data.resize(total_non_zeros);

    // Populate the indices and data arrays for C from the intermediate results.
    // This can be parallelized as well, as each thread would write to a distinct memory location.
    #pragma omp parallel for
    for (int i = 0; i < C.rows; ++i) {
        int offset = C.indptr[i];
        for (size_t j = 0; j < intermediate_rows[i].size(); ++j) {
            C.indices[offset + j] = intermediate_rows[i][j].first;
            C.data[offset + j] = intermediate_rows[i][j].second;
        }
    }
}