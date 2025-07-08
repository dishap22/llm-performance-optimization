#include "csr_io.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <omp.h>
#include <unordered_map>
#include <unordered_set>

// spgemm_cpu.cpp (Optimized Version)
void spgemm_cpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    // Set the dimensions of the output matrix C
    C.rows = A.rows;
    C.cols = B.cols;
    C.indptr.resize(C.rows + 1, 0);

    // --- PASS 1: Symbolic Phase ---
    // Goal: Determine the number of non-zero elements (NNZ) for each row of C.
    // This allows us to pre-allocate the exact amount of memory needed for C's data and indices.

    std::vector<int> row_nnz(C.rows);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < A.rows; ++i) {
        // Use a hash set for efficient tracking of unique column indices for the current output row.
        std::unordered_set<int> unique_cols;

        // Iterate over the non-zero elements of row i in matrix A
        for (int jj = A.indptr[i]; jj < A.indptr[i + 1]; ++jj) {
            int j = A.indices[jj]; // Column index in A (which is the row index in B)

            // Iterate over the non-zero elements of row j in matrix B
            for (int kk = B.indptr[j]; kk < B.indptr[j + 1]; ++kk) {
                int k = B.indices[kk]; // Column index in B
                unique_cols.insert(k);
            }
        }
        row_nnz[i] = unique_cols.size();
    }

    // --- Prefix Sum to calculate C.indptr ---
    // Serially compute the row pointers for C from the NNZ counts.
    for (int i = 0; i < C.rows; ++i) {
        C.indptr[i + 1] = C.indptr[i] + row_nnz[i];
    }

    // Allocate the final storage for C's indices and data arrays.
    int total_non_zeros = C.indptr[C.rows];
    C.indices.resize(total_non_zeros);
    C.data.resize(total_non_zeros);

    // --- PASS 2: Numeric Phase ---
    // Goal: Compute the actual values of C and place them directly into the pre-allocated arrays.

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < A.rows; ++i) {
        // Use a hash map as a sparse accumulator for the current row.
        // This avoids allocating a dense vector of size B.cols.
        std::unordered_map<int, float> accumulator;

        // Re-compute the product for row i
        for (int jj = A.indptr[i]; jj < A.indptr[i + 1]; ++jj) {
            int j = A.indices[jj];
            float val_A = A.data[jj];

            for (int kk = B.indptr[j]; kk < B.indptr[j + 1]; ++kk) {
                int k = B.indices[kk];
                float val_B = B.data[kk];
                accumulator[k] += val_A * val_B;
            }
        }

        // --- Place results directly into C ---
        // Get the starting position for the current row in the final C arrays.
        int offset = C.indptr[i];

        // The accumulator map is unsorted. We need to sort the results by column index
        // before placing them into the CSR structure.
        std::vector<std::pair<int, float>> sorted_elements(accumulator.begin(), accumulator.end());
        std::sort(sorted_elements.begin(), sorted_elements.end(),
                  [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
            return a.first < b.first;
        });

        // Copy the sorted results into their final destination.
        for (size_t j = 0; j < sorted_elements.size(); ++j) {
            C.indices[offset + j] = sorted_elements[j].first;
            C.data[offset + j] = sorted_elements[j].second;
        }
    }
}