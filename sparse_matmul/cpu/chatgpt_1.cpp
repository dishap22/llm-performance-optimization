#include <vector>
#include <unordered_map>
#include <omp.h>
#include "csr_io.h"

void spgemm_cpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    const int A_rows = A.rows;
    const int B_cols = B.cols;

    std::vector<int> C_indptr(A_rows + 1, 0);
    std::vector<std::vector<int>> row_indices(A_rows);
    std::vector<std::vector<float>> row_data(A_rows);

    // Phase 1: symbolic multiplication to determine structure of C
    #pragma omp parallel
    {
        std::unordered_map<int, float> local_map;
        std::vector<std::pair<int, float>> local_entries;

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

            // Move map entries to final row structure
            int count = 0;
            for (const auto& kv : local_map) {
                row_indices[i].push_back(kv.first);
                row_data[i].push_back(kv.second);
                ++count;
            }
            C_indptr[i + 1] = count;
        }
    }

    // Phase 2: finalize indptr
    for (int i = 0; i < A_rows; ++i) {
        C_indptr[i + 1] += C_indptr[i];
    }

    // Phase 3: allocate and fill indices and data
    int nnz = C_indptr.back();
    std::vector<int> C_indices(nnz);
    std::vector<float> C_data(nnz);

    #pragma omp parallel for
    for (int i = 0; i < A_rows; ++i) {
        int offset = C_indptr[i];
        int len = row_indices[i].size();

        std::vector<std::pair<int, float>> entries(len);
        for (int j = 0; j < len; ++j) {
            entries[j] = {row_indices[i][j], row_data[i][j]};
        }

        // Optional: sort by column index for standard CSR
        std::sort(entries.begin(), entries.end());

        for (int j = 0; j < len; ++j) {
            C_indices[offset + j] = entries[j].first;
            C_data[offset + j] = entries[j].second;
        }
    }

    // Assign output
    C.indptr = std::move(C_indptr);
    C.indices = std::move(C_indices);
    C.data = std::move(C_data);
    C.rows = A.rows;
    C.cols = B.cols;
}
