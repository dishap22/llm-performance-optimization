#include <mkl.h>
#include "../utils/csr_io.h"
#include <vector>
#include <cassert>

// C = A * B using Intel MKL
void spgemm_cpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    assert(A.cols == B.rows);

    // Create MKL handles for A and B
    sparse_matrix_t mklA, mklB, mklC;
    struct matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;

    // MKL expects int* for indptr/indices and float* for data
    mkl_sparse_s_create_csr(&mklA, SPARSE_INDEX_BASE_ZERO, A.rows, A.cols,
                            const_cast<int*>(A.indptr.data()),
                            const_cast<int*>(A.indptr.data() + 1),
                            const_cast<int*>(A.indices.data()),
                            const_cast<float*>(A.data.data()));

    mkl_sparse_s_create_csr(&mklB, SPARSE_INDEX_BASE_ZERO, B.rows, B.cols,
                            const_cast<int*>(B.indptr.data()),
                            const_cast<int*>(B.indptr.data() + 1),
                            const_cast<int*>(B.indices.data()),
                            const_cast<float*>(B.data.data()));

    // Perform C = A * B
    mkl_sparse_spmm(SPARSE_OPERATION_NON_TRANSPOSE, mklA, mklB, &mklC);

    // Export C from MKL to CSR arrays
    sparse_index_base_t c_indexing;
    MKL_INT rows, cols, *rows_start, *rows_end, *columns;
    float* values;
    mkl_sparse_s_export_csr(mklC, &c_indexing, &rows, &cols, &rows_start, &rows_end, &columns, &values);

    // Fill CSRMatrix C
    C.rows = rows;
    C.cols = cols;
    C.indptr.resize(rows + 1);
    for (int i = 0; i <= rows; ++i) C.indptr[i] = rows_start[i];
    int nnz = rows_end[rows-1];
    C.indices.assign(columns, columns + nnz);
    C.data.assign(values, values + nnz);

    // Clean up
    mkl_sparse_destroy(mklA);
    mkl_sparse_destroy(mklB);
    mkl_sparse_destroy(mklC);
}
