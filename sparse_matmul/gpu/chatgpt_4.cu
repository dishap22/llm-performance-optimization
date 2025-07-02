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

__global__ void spgemm_count_kernel_dense(
    const int* A_indptr, const int* A_indices,
    const int* B_indptr, const int* B_indices,
    int A_rows, int B_cols, int* C_row_sizes)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) return;

    extern __shared__ int flags[];
    int* flag = flags + threadIdx.x * B_cols;

    for (int i = 0; i < B_cols; ++i)
        flag[i] = 0;

    int count = 0;
    int a_start = A_indptr[row];
    int a_end = A_indptr[row + 1];

    for (int i = a_start; i < a_end; ++i) {
        int a_col = A_indices[i];
        int b_start = B_indptr[a_col];
        int b_end = B_indptr[a_col + 1];

        for (int j = b_start; j < b_end; ++j) {
            int b_col = B_indices[j];
            if (flag[b_col] == 0) {
                flag[b_col] = 1;
                count++;
            }
        }
    }

    C_row_sizes[row] = count;
}

__global__ void spgemm_compute_kernel_dense(
    const int* A_indptr, const int* A_indices, const float* A_data,
    const int* B_indptr, const int* B_indices, const float* B_data,
    const int* C_indptr, int* C_indices, float* C_data,
    int A_rows, int B_cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) return;

    extern __shared__ float shmem[];
    float* values = shmem + threadIdx.x * B_cols;
    int* flags = (int*)(shmem + blockDim.x * B_cols);
    flags += threadIdx.x * B_cols;

    for (int i = 0; i < B_cols; ++i) {
        values[i] = 0.0f;
        flags[i] = 0;
    }

    int a_start = A_indptr[row];
    int a_end = A_indptr[row + 1];

    for (int i = a_start; i < a_end; ++i) {
        int a_col = A_indices[i];
        float a_val = A_data[i];

        int b_start = B_indptr[a_col];
        int b_end = B_indptr[a_col + 1];

        for (int j = b_start; j < b_end; ++j) {
            int b_col = B_indices[j];
            float b_val = B_data[j];

            if (flags[b_col] == 0) flags[b_col] = 1;
            values[b_col] += a_val * b_val;
        }
    }

    int c_start = C_indptr[row];
    int idx = 0;
    for (int i = 0; i < B_cols; ++i) {
        if (flags[i]) {
            C_indices[c_start + idx] = i;
            C_data[c_start + idx] = values[i];
            idx++;
        }
    }
}

void spgemm_gpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    assert(A.cols == B.rows);
    assert(B.cols <= 4096);  // Limit to avoid excessive shared memory

    const int A_rows = A.rows;
    const int B_cols = B.cols;
    const int nnz_A = A.indices.size();
    const int nnz_B = B.indices.size();

    // Allocate and copy A
    int* d_A_indptr; int* d_A_indices; float* d_A_data;
    CHECK_CUDA(cudaMalloc(&d_A_indptr, sizeof(int) * (A.rows + 1)));
    CHECK_CUDA(cudaMalloc(&d_A_indices, sizeof(int) * nnz_A));
    CHECK_CUDA(cudaMalloc(&d_A_data, sizeof(float) * nnz_A));
    CHECK_CUDA(cudaMemcpy(d_A_indptr, A.indptr.data(), sizeof(int) * (A.rows + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_indices, A.indices.data(), sizeof(int) * nnz_A, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_data, A.data.data(), sizeof(float) * nnz_A, cudaMemcpyHostToDevice));

    // Allocate and copy B
    int* d_B_indptr; int* d_B_indices; float* d_B_data;
    CHECK_CUDA(cudaMalloc(&d_B_indptr, sizeof(int) * (B.rows + 1)));
    CHECK_CUDA(cudaMalloc(&d_B_indices, sizeof(int) * nnz_B));
    CHECK_CUDA(cudaMalloc(&d_B_data, sizeof(float) * nnz_B));
    CHECK_CUDA(cudaMemcpy(d_B_indptr, B.indptr.data(), sizeof(int) * (B.rows + 1), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_indices, B.indices.data(), sizeof(int) * nnz_B, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_data, B.data.data(), sizeof(float) * nnz_B, cudaMemcpyHostToDevice));

    // Count nnz per row
    int* d_C_row_sizes;
    CHECK_CUDA(cudaMalloc(&d_C_row_sizes, sizeof(int) * A_rows));

    int blockSize = 128;
    int gridSize = (A_rows + blockSize - 1) / blockSize;
    size_t shmemSize = blockSize * B_cols * sizeof(int);
    spgemm_count_kernel_dense<<<gridSize, blockSize, shmemSize>>>(d_A_indptr, d_A_indices,
                                                                   d_B_indptr, d_B_indices,
                                                                   A_rows, B_cols, d_C_row_sizes);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Exclusive scan to get C.indptr
    std::vector<int> h_row_sizes(A_rows);
    CHECK_CUDA(cudaMemcpy(h_row_sizes.data(), d_C_row_sizes, sizeof(int) * A_rows, cudaMemcpyDeviceToHost));

    C.indptr.resize(A_rows + 1, 0);
    for (int i = 0; i < A_rows; ++i)
        C.indptr[i + 1] = C.indptr[i] + h_row_sizes[i];
    int total_nnz = C.indptr.back();
    C.indices.resize(total_nnz);
    C.data.resize(total_nnz);

    // Allocate device memory for C
    int* d_C_indptr; int* d_C_indices; float* d_C_data;
    CHECK_CUDA(cudaMalloc(&d_C_indptr, sizeof(int) * (A_rows + 1)));
    CHECK_CUDA(cudaMalloc(&d_C_indices, sizeof(int) * total_nnz));
    CHECK_CUDA(cudaMalloc(&d_C_data, sizeof(float) * total_nnz));
    CHECK_CUDA(cudaMemcpy(d_C_indptr, C.indptr.data(), sizeof(int) * (A_rows + 1), cudaMemcpyHostToDevice));

    // Compute actual C values
    shmemSize = blockSize * B_cols * sizeof(float) + blockSize * B_cols * sizeof(int);
    spgemm_compute_kernel_dense<<<gridSize, blockSize, shmemSize>>>(
        d_A_indptr, d_A_indices, d_A_data,
        d_B_indptr, d_B_indices, d_B_data,
        d_C_indptr, d_C_indices, d_C_data,
        A_rows, B_cols);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(C.indices.data(), d_C_indices, sizeof(int) * total_nnz, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C.data.data(), d_C_data, sizeof(float) * total_nnz, cudaMemcpyDeviceToHost));

    C.rows = A.rows;
    C.cols = B.cols;

    // Cleanup
    cudaFree(d_A_indptr); cudaFree(d_A_indices); cudaFree(d_A_data);
    cudaFree(d_B_indptr); cudaFree(d_B_indices); cudaFree(d_B_data);
    cudaFree(d_C_row_sizes); cudaFree(d_C_indptr); cudaFree(d_C_indices); cudaFree(d_C_data);
}
