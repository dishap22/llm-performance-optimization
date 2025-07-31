#include <vector>
#include <algorithm>
#include <omp.h>
#include <numeric> // for std::partial_sum
#include <cstring> // for memset
#include "csr_io.h"

void spgemm_cpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    const int A_rows = A.rows;
    const int B_cols = B.cols;

    std::vector<int> C_indptr(A_rows + 1, 0);
    std::vector<std::vector<int>> row_indices(A_rows);
    std::vector<std::vector<float>> row_data(A_rows);

    #pragma omp parallel
    {
        std::vector<float> accumulator(B_cols, 0.0f);
        std::vector<char> mask(B_cols, 0);
        std::vector<int> touched_cols;
        touched_cols.reserve(256); // heuristic

        #pragma omp for
        for (int i = 0; i < A_rows; ++i) {
            touched_cols.clear();

            int row_start = A.indptr[i];
            int row_end = A.indptr[i + 1];

            for (int jj = row_start; jj < row_end; ++jj) {
                int a_col = A.indices[jj];
                float a_val = A.data[jj];

                int b_row_start = B.indptr[a_col];
                int b_row_end = B.indptr[a_col + 1];

                for (int kk = b_row_start; kk < b_row_end; ++kk) {
                    int b_col = B.indices[kk];
                    float b_val = B.data[kk];

                    if (!mask[b_col]) {
                        mask[b_col] = 1;
                        touched_cols.push_back(b_col);
                    }
                    accumulator[b_col] += a_val * b_val;
                }
            }

            int nnz = static_cast<int>(touched_cols.size());
            row_indices[i].reserve(nnz);
            row_data[i].reserve(nnz);

            for (int col : touched_cols) {
                row_indices[i].push_back(col);
                row_data[i].push_back(accumulator[col]);

                // Reset accumulator and mask
                accumulator[col] = 0.0f;
                mask[col] = 0;
            }

            C_indptr[i + 1] = nnz;
        }
    }

    // Exclusive prefix sum to finalize indptr
    std::partial_sum(C_indptr.begin(), C_indptr.end(), C_indptr.begin());

    int nnz = C_indptr.back();
    std::vector<int> C_indices(nnz);
    std::vector<float> C_data(nnz);

    #pragma omp parallel for
    for (int i = 0; i < A_rows; ++i) {
        int offset = C_indptr[i];
        int len = static_cast<int>(row_indices[i].size());

        std::vector<std::pair<int, float>> entries(len);
        for (int j = 0; j < len; ++j) {
            entries[j] = {row_indices[i][j], row_data[i][j]};
        }

        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        for (int j = 0; j < len; ++j) {
            C_indices[offset + j] = entries[j].first;
            C_data[offset + j] = entries[j].second;
        }
    }

    // Final CSR assignment
    C.indptr = std::move(C_indptr);
    C.indices = std::move(C_indices);
    C.data = std::move(C_data);
    C.rows = A.rows;
    C.cols = B.cols;
}
