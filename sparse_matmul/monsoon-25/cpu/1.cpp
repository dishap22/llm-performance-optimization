#include "spgemm_interface.h"
#include <vector>
#include <omp.h>
#include <unordered_map>
#include <algorithm>
#include <stdexcept>

void spgemm_cpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    if (A.cols != B.rows) {
        throw std::invalid_argument("Matrix dimensions are not compatible for multiplication.");
    }

    const int num_rows_A = A.rows;
    const int num_cols_C = B.cols;

    C.rows = num_rows_A;
    C.cols = num_cols_C;

    C.indptr.assign(num_rows_A + 1, 0);

    // Phase 1: Symbolic Multiplication - Count non-zero elements in each row of C
    std::vector<int> c_row_nnz(num_rows_A, 0);
    #pragma omp parallel for
    for (int i = 0; i < num_rows_A; ++i) {
        std::unordered_map<int, double> temp_row;
        for (int j = A.indptr[i]; j < A.indptr[i+1]; ++j) {
            int col_A = A.indices[j];
            double val_A = A.data[j];

            for (int k = B.indptr[col_A]; k < B.indptr[col_A+1]; ++k) {
                int col_B = B.indices[k];
                double val_B = B.data[k];
                temp_row[col_B] += val_A * val_B;
            }
        }
        c_row_nnz[i] = temp_row.size();
    }

    // Prefix sum to calculate C.indptr
    for (int i = 0; i < num_rows_A; ++i) {
        C.indptr[i+1] = C.indptr[i] + c_row_nnz[i];
    }
    C.indices.resize(C.indptr.back());
    C.data.resize(C.indptr.back());

    // Phase 2: Numeric Multiplication - Compute values and indices of C
    #pragma omp parallel for
    for (int i = 0; i < num_rows_A; ++i) {
        std::unordered_map<int, double> temp_row;
        for (int j = A.indptr[i]; j < A.indptr[i+1]; ++j) {
            int col_A = A.indices[j];
            double val_A = A.data[j];

            for (int k = B.indptr[col_A]; k < B.indptr[col_A+1]; ++k) {
                int col_B = B.indices[k];
                double val_B = B.data[k];
                temp_row[col_B] += val_A * val_B;
            }
        }

        std::vector<std::pair<int, double>> sorted_row(temp_row.begin(), temp_row.end());
        std::sort(sorted_row.begin(), sorted_row.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });

        int current_pos = C.indptr[i];
        for (const auto& pair : sorted_row) {
            C.indices[current_pos] = pair.first;
            C.data[current_pos] = pair.second;
            current_pos++;
        }
    }
}