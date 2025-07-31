// chatgpt_optimized.cu
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

constexpr int THREADS_PER_ROW = 32;

__global__ void spgemm_count_kernel_optimized(
    const int* __restrict__ A_indptr, const int* __restrict__ A_indices,
    const int* __restrict__ B_indptr, const int* __restrict__ B_indices,
    int A_rows, int B_cols,
    int* __restrict__ row_nnz, int* __restrict__ mask)
{
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int row = global_thread_id / THREADS_PER_ROW;
    int lane = threadIdx.x % THREADS_PER_ROW;
    if (row >= A_rows) return;

    int* row_mask = mask + row * B_cols;

    // Initialize mask
    for (int i = lane; i < B_cols; i += THREADS_PER_ROW)
        row_mask[i] = 0;

    __syncthreads();

    for (int ai = A_indptr[row]; ai < A_indptr[row + 1]; ++ai) {
        int a_col = A_indices[ai];
        for (int bi = B_indptr[a_col] + lane; bi < B_indptr[a_col + 1]; bi += THREADS_PER_ROW) {
            int b_col = B_indices[bi];
            atomicExch(&row_mask[b_col], 1);
        }
    }

    __syncthreads();

    int nnz_local = 0;
    for (int i = lane; i < B_cols; i += THREADS_PER_ROW) {
        if (row_mask[i]) ++nnz_local;
    }

    __shared__ int warp_counts[THREADS_PER_ROW];
    warp_counts[lane] = nnz_local;
    __syncthreads();

    // Reduce within warp
    if (lane == 0) {
        int sum = 0;
        for (int i = 0; i < THREADS_PER_ROW; ++i)
            sum += warp_counts[i];
        row_nnz[row] = sum;
    }
}

__global__ void spgemm_compute_kernel_optimized(
    const int* __restrict__ A_indptr, const int* __restrict__ A_indices, const float* __restrict__ A_data,
    const int* __restrict__ B_indptr, const int* __restrict__ B_indices, const float* __restrict__ B_data,
    const int* __restrict__ C_indptr, int* __restrict__ C_indices, float* __restrict__ C_data,
    int A_rows, int B_cols, float* temp_values, int* temp_flags)
{
    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int row = global_thread_id / THREADS_PER_ROW;
    int lane = threadIdx.x % THREADS_PER_ROW;
    if (row >= A_rows) return;

    float* values = temp_values + row * B_cols;
    int* flags = temp_flags + row * B_cols;

    for (int i = lane; i < B_cols; i += THREADS_PER_ROW) {
        values[i] = 0.0f;
        flags[i] = 0;
    }

    __syncthreads();

    for (int ai = A_indptr[row]; ai < A_indptr[row + 1]; ++ai) {
        int a_col = A_indices[ai];
        float a_val = A_data[ai];

        for (int bi = B_indptr[a_col] + lane; bi < B_indptr[a_col + 1]; bi += THREADS_PER_ROW) {
            int b_col = B_indices[bi];
            float b_val = B_data[bi];
            atomicAdd(&values[b_col], a_val * b_val);
            flags[b_col] = 1;
        }
    }

    __syncthreads();

    int start = C_indptr[row];
    int idx_local = 0;
    for (int col = lane; col < B_cols; col += THREADS_PER_ROW) {
        if (flags[col]) {
            int pos = atomicAdd(&idx_local, 1);
            C_indices[start + pos] = col;
            C_data[start + pos] = values[col];
        }
    }
}

void spgemm_gpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    assert(A.cols == B.rows);
    int A_rows = A.rows;
    int B_cols = B.cols;
    int nnz_A = A.indices.size();
    int nnz_B = B.indices.size();

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

    int* d_row_nnz;
    CHECK_CUDA(cudaMalloc(&d_row_nnz, sizeof(int) * A_rows));
    int* d_mask;
    CHECK_CUDA(cudaMalloc(&d_mask, sizeof(int) * A_rows * B_cols));

    // Optimize launch: use warps per row
    int total_threads = A_rows * THREADS_PER_ROW;
    int blockSize = 256;
    int gridSize = (total_threads + blockSize - 1) / blockSize;

    spgemm_count_kernel_optimized<<<gridSize, blockSize>>>(d_A_indptr, d_A_indices,
                                                           d_B_indptr, d_B_indices,
                                                           A_rows, B_cols, d_row_nnz, d_mask);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<int> row_nnz(A_rows);
    CHECK_CUDA(cudaMemcpy(row_nnz.data(), d_row_nnz, sizeof(int) * A_rows, cudaMemcpyDeviceToHost));
    C.indptr.resize(A_rows + 1);
    C.indptr[0] = 0;
    for (int i = 0; i < A_rows; ++i)
        C.indptr[i + 1] = C.indptr[i] + row_nnz[i];

    int nnz_C = C.indptr[A_rows];
    C.indices.resize(nnz_C);
    C.data.resize(nnz_C);

    int* d_C_indptr, *d_C_indices;
    float* d_C_data;
    CHECK_CUDA(cudaMalloc(&d_C_indptr, sizeof(int) * (A_rows + 1)));
    CHECK_CUDA(cudaMalloc(&d_C_indices, sizeof(int) * nnz_C));
    CHECK_CUDA(cudaMalloc(&d_C_data, sizeof(float) * nnz_C));
    CHECK_CUDA(cudaMemcpy(d_C_indptr, C.indptr.data(), sizeof(int) * (A_rows + 1), cudaMemcpyHostToDevice));

    float* d_temp_values;
    int* d_temp_flags;
    CHECK_CUDA(cudaMalloc(&d_temp_values, sizeof(float) * A_rows * B_cols));
    CHECK_CUDA(cudaMalloc(&d_temp_flags, sizeof(int) * A_rows * B_cols));

    spgemm_compute_kernel_optimized<<<gridSize, blockSize>>>(
        d_A_indptr, d_A_indices, d_A_data,
        d_B_indptr, d_B_indices, d_B_data,
        d_C_indptr, d_C_indices, d_C_data,
        A_rows, B_cols, d_temp_values, d_temp_flags);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(C.indices.data(), d_C_indices, sizeof(int) * nnz_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C.data.data(), d_C_data, sizeof(float) * nnz_C, cudaMemcpyDeviceToHost));
    C.rows = A.rows;
    C.cols = B.cols;

    cudaFree(d_A_indptr); cudaFree(d_A_indices); cudaFree(d_A_data);
    cudaFree(d_B_indptr); cudaFree(d_B_indices); cudaFree(d_B_data);
    cudaFree(d_row_nnz); cudaFree(d_mask);
    cudaFree(d_C_indptr); cudaFree(d_C_indices); cudaFree(d_C_data);
    cudaFree(d_temp_values); cudaFree(d_temp_flags);
}
