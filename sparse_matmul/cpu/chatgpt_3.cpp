#include <vector>
#include <unordered_map>
#include <algorithm>
#include <omp.h>
#include <numeric> // for std::partial_sum
#include "csr_io.h"

void spgemm_cpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    const int A_rows = A.rows;

    std::vector<int> C_indptr(A_rows + 1, 0);
    std::vector<std::vector<int>> row_indices(A_rows);
    std::vector<std::vector<float>> row_data(A_rows);

    // Phase 1: symbolic + numeric multiplication (row-wise)
    #pragma omp parallel
    {
        std::vector<std::pair<int, float>> local_accum;
        local_accum.reserve(128);  // heuristic for performance

        std::unordered_map<int, float> local_map;

        #pragma omp for
        for (int i = 0; i < A_rows; ++i) {
            local_map.clear();
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
                    local_map[b_col] += a_val * b_val;
                }
            }

            // Pre-size vectors for efficiency
            row_indices[i].reserve(local_map.size());
            row_data[i].reserve(local_map.size());

            for (const auto& kv : local_map) {
                row_indices[i].push_back(kv.first);
                row_data[i].push_back(kv.second);
            }
            C_indptr[i + 1] = static_cast<int>(local_map.size());
        }
    }

    // Phase 2: exclusive prefix sum for C.indptr
    std::partial_sum(C_indptr.begin(), C_indptr.end(), C_indptr.begin());

    // Phase 3: finalize C.indices and C.data
    int nnz = C_indptr.back();
    std::vector<int> C_indices(nnz);
    std::vector<float> C_data(nnz);

    #pragma omp parallel for
    for (int i = 0; i < A_rows; ++i) {
        int offset = C_indptr[i];
        int len = static_cast<int>(row_indices[i].size());

        // Use zip-sort to avoid copying
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

    // Final CSRMatrix assignment
    C.indptr = std::move(C_indptr);
    C.indices = std::move(C_indices);
    C.data = std::move(C_data);
    C.rows = A.rows;
    C.cols = B.cols;
}
