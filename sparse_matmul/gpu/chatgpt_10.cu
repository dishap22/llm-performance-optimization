// chatgpt_fixed.cu
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

// Define warp size
#define WARP_SIZE 32

// Cooperative count kernel: one warp per row
__global__ void spgemm_count_kernel(
    const int* __restrict__ A_indptr, const int* __restrict__ A_indices,
    const int* __restrict__ B_indptr, const int* __restrict__ B_indices,
    int A_rows, int B_cols, int* __restrict__ row_nnz, int* __restrict__ mask)
{
    int warp_id = blockIdx.x * blockDim.x / WARP_SIZE + threadIdx.x / WARP_SIZE;
    if (warp_id >= A_rows) return;

    int lane = threadIdx.x % WARP_SIZE;
    int row = warp_id;

    int* row_mask = mask + row * B_cols;

    for (int i = lane; i < B_cols; i += WARP_SIZE) row_mask[i] = 0;
    __syncwarp();

    int row_start = A_indptr[row];
    int row_end = A_indptr[row + 1];

    for (int i = row_start + lane; i < row_end; i += WARP_SIZE) {
        if (i < row_end) {
            int a_col = A_indices[i];
            int b_start = B_indptr[a_col];
            int b_end = B_indptr[a_col + 1];
            for (int j = b_start; j < b_end; ++j) {
                int b_col = B_indices[j];
                if (atomicExch(&row_mask[b_col], 1) == 0) {
                    atomicAdd(&row_nnz[row], 1);
                }
            }
        }
    }
}

// Cooperative compute kernel: one warp per row
__global__ void spgemm_compute_kernel(
    const int* __restrict__ A_indptr, const int* __restrict__ A_indices, const float* __restrict__ A_data,
    const int* __restrict__ B_indptr, const int* __restrict__ B_indices, const float* __restrict__ B_data,
    const int* __restrict__ C_indptr, int* __restrict__ C_indices, float* __restrict__ C_data,
    int A_rows, int B_cols, float* temp_values, int* temp_flags)
{
    int warp_id = blockIdx.x * blockDim.x / WARP_SIZE + threadIdx.x / WARP_SIZE;
    if (warp_id >= A_rows) return;

    int lane = threadIdx.x % WARP_SIZE;
    int row = warp_id;

    float* values = temp_values + row * B_cols;
    int* flags = temp_flags + row * B_cols;

    for (int i = lane; i < B_cols; i += WARP_SIZE) {
        values[i] = 0.0f;
        flags[i] = 0;
    }
    __syncwarp();

    int row_start = A_indptr[row];
    int row_end = A_indptr[row + 1];

    for (int i = row_start + lane; i < row_end; i += WARP_SIZE) {
        if (i < row_end) {
            int a_col = A_indices[i];
            float a_val = A_data[i];
            int b_start = B_indptr[a_col];
            int b_end = B_indptr[a_col + 1];
            for (int j = b_start; j < b_end; ++j) {
                int b_col = B_indices[j];
                float b_val = B_data[j];
                atomicAdd(&values[b_col], a_val * b_val);
                flags[b_col] = 1;
            }
        }
    }
    __syncwarp();

    // Write final output
    int out_start = C_indptr[row];
    if (lane == 0) {
        int idx = 0;
        for (int col = 0; col < B_cols; ++col) {
            if (flags[col]) {
                C_indices[out_start + idx] = col;
                C_data[out_start + idx] = values[col];
                ++idx;
            }
        }
    }
}

void spgemm_gpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    assert(A.cols == B.rows);

    int A_rows = A.rows;
    int B_cols = B.cols;
    int nnz_A = A.indices.size();
    int nnz_B = B.indices.size();

    // Copy A and B to device
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
    CHECK_CUDA(cudaMemset(d_row_nnz, 0, sizeof(int) * A_rows));
    int* d_mask;
    CHECK_CUDA(cudaMalloc(&d_mask, sizeof(int) * A_rows * B_cols));
    CHECK_CUDA(cudaMemset(d_mask, 0, sizeof(int) * A_rows * B_cols));

    // Launch count kernel (warp-per-row)
    int warps_per_block = 4;
    int threads_per_block = warps_per_block * WARP_SIZE;
    int gridSize = (A_rows + warps_per_block - 1) / warps_per_block;
    spgemm_count_kernel<<<gridSize, threads_per_block>>>(d_A_indptr, d_A_indices,
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
    int* d_temp_flags;
    CHECK_CUDA(cudaMalloc(&d_temp_values, sizeof(float) * A_rows * B_cols));
    CHECK_CUDA(cudaMalloc(&d_temp_flags, sizeof(int) * A_rows * B_cols));
    CHECK_CUDA(cudaMemset(d_temp_values, 0, sizeof(float) * A_rows * B_cols));
    CHECK_CUDA(cudaMemset(d_temp_flags, 0, sizeof(int) * A_rows * B_cols));

    // Compute kernel (warp-per-row)
    spgemm_compute_kernel<<<gridSize, threads_per_block>>>(
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
