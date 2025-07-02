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

// Count kernel
__global__ void spgemm_count_kernel(const int* A_indptr, const int* A_indices,
                                    const int* B_indptr, const int* B_indices,
                                    int A_rows, int B_cols,
                                    int* C_row_sizes) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) return;

    extern __shared__ int shmem_int[];
    int* mask = shmem_int + threadIdx.x * B_cols;

    for (int i = 0; i < B_cols; ++i) mask[i] = -1;

    int start = A_indptr[row];
    int end = A_indptr[row + 1];

    int count = 0;

    for (int i = start; i < end; ++i) {
        int a_col = A_indices[i];

        int b_start = B_indptr[a_col];
        int b_end = B_indptr[a_col + 1];

        for (int j = b_start; j < b_end; ++j) {
            int b_col = B_indices[j];
            if (mask[b_col] != row) {
                mask[b_col] = row;
                count++;
            }
        }
    }

    C_row_sizes[row] = count;
}

// Compute kernel
__global__ void spgemm_compute_kernel(const int* A_indptr, const int* A_indices, const float* A_data,
                                      const int* B_indptr, const int* B_indices, const float* B_data,
                                      const int* C_indptr, int* C_indices, float* C_data,
                                      int A_rows, int B_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) return;

    extern __shared__ float shmem[]; // shared buffer for both vals and flags

    float* vals = shmem + threadIdx.x * B_cols;
    int* flags = (int*)(shmem + blockDim.x * B_cols);
    flags += threadIdx.x * B_cols;

    for (int i = 0; i < B_cols; ++i) {
        vals[i] = 0.0f;
        flags[i] = 0;
    }

    int start = A_indptr[row];
    int end = A_indptr[row + 1];

    for (int i = start; i < end; ++i) {
        int a_col = A_indices[i];
        float a_val = A_data[i];

        int b_start = B_indptr[a_col];
        int b_end = B_indptr[a_col + 1];

        for (int j = b_start; j < b_end; ++j) {
            int b_col = B_indices[j];
            float b_val = B_data[j];
            if (!flags[b_col]) {
                flags[b_col] = 1;
            }
            vals[b_col] += a_val * b_val;
        }
    }

    int c_start = C_indptr[row];
    int idx = 0;

    for (int i = 0; i < B_cols; ++i) {
        if (flags[i]) {
            C_indices[c_start + idx] = i;
            C_data[c_start + idx] = vals[i];
            idx++;
        }
    }
}

// Main SpGEMM function
void spgemm_gpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    assert(A.cols == B.rows);
    assert(B.cols < 4096); // Limit for shared memory usage

    int A_rows = A.rows;
    int B_cols = B.cols;

    int nnz_A = A.indices.size();
    int nnz_B = B.indices.size();

    // Allocate device memory for A
    int* d_A_indptr; int* d_A_indices; float* d_A_data;
    CHECK_CUDA(cudaMalloc(&d_A_indptr, (A.rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_A_indices, nnz_A * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_A_data, nnz_A * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A_indptr, A.indptr.data(), (A.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_indices, A.indices.data(), nnz_A * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_data, A.data.data(), nnz_A * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate device memory for B
    int* d_B_indptr; int* d_B_indices; float* d_B_data;
    CHECK_CUDA(cudaMalloc(&d_B_indptr, (B.rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_B_indices, nnz_B * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_B_data, nnz_B * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_B_indptr, B.indptr.data(), (B.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_indices, B.indices.data(), nnz_B * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_data, B.data.data(), nnz_B * sizeof(float), cudaMemcpyHostToDevice));

    // Count non-zeros per row in C
    int* d_C_row_sizes;
    CHECK_CUDA(cudaMalloc(&d_C_row_sizes, A_rows * sizeof(int)));

    int blockSize = 128;
    int gridSize = (A_rows + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * B_cols * sizeof(int);
    spgemm_count_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_A_indptr, d_A_indices,
                                                                 d_B_indptr, d_B_indices,
                                                                 A_rows, B_cols, d_C_row_sizes);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int> h_C_row_sizes(A_rows);
    CHECK_CUDA(cudaMemcpy(h_C_row_sizes.data(), d_C_row_sizes, A_rows * sizeof(int), cudaMemcpyDeviceToHost));

    // Compute C.indptr
    C.indptr.resize(A_rows + 1, 0);
    for (int i = 0; i < A_rows; ++i)
        C.indptr[i + 1] = C.indptr[i] + h_C_row_sizes[i];

    int total_nnz = C.indptr.back();
    C.indices.resize(total_nnz);
    C.data.resize(total_nnz);

    // Copy indptr to device
    int* d_C_indptr;
    CHECK_CUDA(cudaMalloc(&d_C_indptr, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_C_indptr, C.indptr.data(), (A_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate memory for output
    int* d_C_indices; float* d_C_data;
    CHECK_CUDA(cudaMalloc(&d_C_indices, total_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_C_data, total_nnz * sizeof(float)));

    sharedMemSize = 2 * blockSize * B_cols * sizeof(float); // vals + flags
    spgemm_compute_kernel<<<gridSize, blockSize, sharedMemSize>>>(
        d_A_indptr, d_A_indices, d_A_data,
        d_B_indptr, d_B_indices, d_B_data,
        d_C_indptr, d_C_indices, d_C_data,
        A_rows, B_cols);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(C.indices.data(), d_C_indices, total_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C.data.data(), d_C_data, total_nnz * sizeof(float), cudaMemcpyDeviceToHost));

    C.rows = A.rows;
    C.cols = B.cols;

    // Cleanup
    cudaFree(d_A_indptr); cudaFree(d_A_indices); cudaFree(d_A_data);
    cudaFree(d_B_indptr); cudaFree(d_B_indices); cudaFree(d_B_data);
    cudaFree(d_C_row_sizes); cudaFree(d_C_indptr);
    cudaFree(d_C_indices); cudaFree(d_C_data);
}
