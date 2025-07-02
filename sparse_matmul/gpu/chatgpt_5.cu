// chatgpt_1.cu
#include "spgemm_interface.h"
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include <cstdio>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Count non-zero columns in each output row
__global__ void spgemm_count_kernel(
    const int* A_indptr, const int* A_indices,
    const int* B_indptr, const int* B_indices,
    int A_rows, int B_cols, int* row_nnz, uint8_t* mask)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) return;

    uint8_t* row_mask = mask + row * B_cols;
    for (int i = 0; i < B_cols; ++i) row_mask[i] = 0;

    int nnz = 0;
    for (int ai = A_indptr[row]; ai < A_indptr[row + 1]; ++ai) {
        int a_col = A_indices[ai];
        for (int bi = B_indptr[a_col]; bi < B_indptr[a_col + 1]; ++bi) {
            int b_col = B_indices[bi];
            if (row_mask[b_col] == 0) {
                row_mask[b_col] = 1;
                nnz++;
            }
        }
    }
    row_nnz[row] = nnz;
}

// Compute actual values into C
__global__ void spgemm_compute_kernel(
    const int* A_indptr, const int* A_indices, const float* A_data,
    const int* B_indptr, const int* B_indices, const float* B_data,
    const int* C_indptr, int* C_indices, float* C_data,
    int A_rows, int B_cols, float* temp_values, uint8_t* temp_flags)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) return;

    float* values = temp_values + row * B_cols;
    uint8_t* flags = temp_flags + row * B_cols;
    for (int i = 0; i < B_cols; ++i) {
        values[i] = 0.0f;
        flags[i] = 0;
    }

    for (int ai = A_indptr[row]; ai < A_indptr[row + 1]; ++ai) {
        int a_col = A_indices[ai];
        float a_val = A_data[ai];
        for (int bi = B_indptr[a_col]; bi < B_indptr[a_col + 1]; ++bi) {
            int b_col = B_indices[bi];
            float b_val = B_data[bi];
            values[b_col] += a_val * b_val;
            flags[b_col] = 1;
        }
    }

    int idx = 0;
    int start = C_indptr[row];
    for (int col = 0; col < B_cols; ++col) {
        if (flags[col]) {
            C_indices[start + idx] = col;
            C_data[start + idx] = values[col];
            idx++;
        }
    }
}

void spgemm_gpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    assert(A.cols == B.rows);

    int A_rows = A.rows;
    int B_cols = B.cols;
    int nnz_A = A.indices.size();
    int nnz_B = B.indices.size();

    // --- Copy A and B to device ---
    int *d_A_indptr, *d_A_indices, *d_B_indptr, *d_B_indices;
    float *d_A_data, *d_B_data;
    CHECK_CUDA(cudaMalloc(&d_A_indptr, sizeof(int) * (A.rows + 1)));
    CHECK_CUDA(cudaMalloc(&d_A_indices, sizeof(int) * nnz_A));
    CHECK_CUDA(cudaMalloc(&d_A_data, sizeof(float) * nnz_A));
    CHECK_CUDA(cudaMemcpy(d_A_indptr, A.indptr.data(), sizeof(int) * (A.rows + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_indices, A.indices.data(), sizeof(int) * nnz_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_data, A.data.data(), sizeof(float) * nnz_A, cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&d_B_indptr, sizeof(int) * (B.rows + 1)));
    CHECK_CUDA(cudaMalloc(&d_B_indices, sizeof(int) * nnz_B));
    CHECK_CUDA(cudaMalloc(&d_B_data, sizeof(float) * nnz_B));
    CHECK_CUDA(cudaMemcpy(d_B_indptr, B.indptr.data(), sizeof(int) * (B.rows + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_indices, B.indices.data(), sizeof(int) * nnz_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_data, B.data.data(), sizeof(float) * nnz_B, cudaMemcpyHostToDevice));

    // Allocate row nnz count and masks
    int* d_row_nnz;
    CHECK_CUDA(cudaMalloc(&d_row_nnz, sizeof(int) * A_rows));
    uint8_t* d_mask;
    CHECK_CUDA(cudaMalloc(&d_mask, sizeof(uint8_t) * A_rows * B_cols));

    // Launch count kernel
    int blockSize = 128;
    int gridSize = (A_rows + blockSize - 1) / blockSize;
    spgemm_count_kernel<<<gridSize, blockSize>>>(d_A_indptr, d_A_indices,
                                                 d_B_indptr, d_B_indices,
                                                 A_rows, B_cols, d_row_nnz, d_mask);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Download row sizes and build indptr
    std::vector<int> row_nnz(A_rows);
    CHECK_CUDA(cudaMemcpy(row_nnz.data(), d_row_nnz, sizeof(int) * A_rows, cudaMemcpyDeviceToHost));
    C.indptr.resize(A_rows + 1);
    C.indptr[0] = 0;
    for (int i = 0; i < A_rows; ++i)
        C.indptr[i + 1] = C.indptr[i] + row_nnz[i];

    int nnz_C = C.indptr[A_rows];
    C.indices.resize(nnz_C);
    C.data.resize(nnz_C);

    // Allocate output on device
    int* d_C_indptr, *d_C_indices;
    float* d_C_data;
    CHECK_CUDA(cudaMalloc(&d_C_indptr, sizeof(int) * (A_rows + 1)));
    CHECK_CUDA(cudaMalloc(&d_C_indices, sizeof(int) * nnz_C));
    CHECK_CUDA(cudaMalloc(&d_C_data, sizeof(float) * nnz_C));
    CHECK_CUDA(cudaMemcpy(d_C_indptr, C.indptr.data(), sizeof(int) * (A_rows + 1), cudaMemcpyHostToDevice));

    // Temp buffers
    float* d_temp_values;
    uint8_t* d_temp_flags;
    CHECK_CUDA(cudaMalloc(&d_temp_values, sizeof(float) * A_rows * B_cols));
    CHECK_CUDA(cudaMalloc(&d_temp_flags, sizeof(uint8_t) * A_rows * B_cols));

    // Compute kernel
    spgemm_compute_kernel<<<gridSize, blockSize>>>(
        d_A_indptr, d_A_indices, d_A_data,
        d_B_indptr, d_B_indices, d_B_data,
        d_C_indptr, d_C_indices, d_C_data,
        A_rows, B_cols, d_temp_values, d_temp_flags);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Download C
    CHECK_CUDA(cudaMemcpy(C.indices.data(), d_C_indices, sizeof(int) * nnz_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C.data.data(), d_C_data, sizeof(float) * nnz_C, cudaMemcpyDeviceToHost));
    C.rows = A.rows;
    C.cols = B.cols;

    // Free memory
    cudaFree(d_A_indptr); cudaFree(d_A_indices); cudaFree(d_A_data);
    cudaFree(d_B_indptr); cudaFree(d_B_indices); cudaFree(d_B_data);
    cudaFree(d_row_nnz); cudaFree(d_mask);
    cudaFree(d_C_indptr); cudaFree(d_C_indices); cudaFree(d_C_data);
    cudaFree(d_temp_values); cudaFree(d_temp_flags);
}
