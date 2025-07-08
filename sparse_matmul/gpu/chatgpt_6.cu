// chatgpt_optimized_spgemm.cu
#include "spgemm_interface.h"
#include <cuda_runtime.h>
#include <cassert>
#include <vector>
#include <cstdio>
#include <unordered_map>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Count non-zero columns using a small fixed-size hash table
__global__ void spgemm_count_kernel_opt(
    const int* __restrict__ A_indptr, const int* __restrict__ A_indices,
    const int* __restrict__ B_indptr, const int* __restrict__ B_indices,
    int A_rows, int B_cols, int* row_nnz)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) return;

    constexpr int HASH_SIZE = 1024;
    __shared__ int hash_table[HASH_SIZE];
    __shared__ uint8_t hash_flag[HASH_SIZE];

    for (int i = threadIdx.x; i < HASH_SIZE; i += blockDim.x) {
        hash_table[i] = -1;
        hash_flag[i] = 0;
    }
    __syncthreads();

    int count = 0;
    for (int ai = A_indptr[row]; ai < A_indptr[row + 1]; ++ai) {
        int a_col = A_indices[ai];
        for (int bi = B_indptr[a_col]; bi < B_indptr[a_col + 1]; ++bi) {
            int b_col = B_indices[bi];
            int hash = b_col % HASH_SIZE;

            while (true) {
                int old = atomicCAS(&hash_table[hash], -1, b_col);
                if (old == -1 || old == b_col) {
                    if (atomicExch(&hash_flag[hash], 1) == 0) {
                        ++count;
                    }
                    break;
                }
                hash = (hash + 1) % HASH_SIZE;
            }
        }
    }

    row_nnz[row] = count;
}

// Actual SpGEMM compute using float accumulator and index hashing
__global__ void spgemm_compute_kernel_opt(
    const int* __restrict__ A_indptr, const int* __restrict__ A_indices, const float* __restrict__ A_data,
    const int* __restrict__ B_indptr, const int* __restrict__ B_indices, const float* __restrict__ B_data,
    const int* __restrict__ C_indptr, int* C_indices, float* C_data,
    int A_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) return;

    constexpr int HASH_SIZE = 2048;
    __shared__ int hash_keys[HASH_SIZE];
    __shared__ float hash_vals[HASH_SIZE];
    __shared__ uint8_t hash_flags[HASH_SIZE];

    for (int i = threadIdx.x; i < HASH_SIZE; i += blockDim.x) {
        hash_keys[i] = -1;
        hash_vals[i] = 0.0f;
        hash_flags[i] = 0;
    }
    __syncthreads();

    for (int ai = A_indptr[row]; ai < A_indptr[row + 1]; ++ai) {
        int a_col = A_indices[ai];
        float a_val = A_data[ai];

        for (int bi = B_indptr[a_col]; bi < B_indptr[a_col + 1]; ++bi) {
            int b_col = B_indices[bi];
            float b_val = B_data[bi];
            float product = a_val * b_val;

            int hash = b_col % HASH_SIZE;
            while (true) {
                int old = atomicCAS(&hash_keys[hash], -1, b_col);
                if (old == -1 || old == b_col) {
                    atomicAdd(&hash_vals[hash], product);
                    hash_flags[hash] = 1;
                    break;
                }
                hash = (hash + 1) % HASH_SIZE;
            }
        }
    }

    int start = C_indptr[row];
    int idx = 0;
    for (int i = 0; i < HASH_SIZE && idx < (C_indptr[row + 1] - start); ++i) {
        if (hash_flags[i]) {
            C_indices[start + idx] = hash_keys[i];
            C_data[start + idx] = hash_vals[i];
            ++idx;
        }
    }
}

void spgemm_gpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    assert(A.cols == B.rows);

    int A_rows = A.rows;
    int B_cols = B.cols;
    int nnz_A = A.indices.size();
    int nnz_B = B.indices.size();

    // --- Device Memory Allocation ---
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

    // Count non-zeros per row in C
    int* d_row_nnz;
    CHECK_CUDA(cudaMalloc(&d_row_nnz, sizeof(int) * A_rows));

    int blockSize = 128;
    int gridSize = (A_rows + blockSize - 1) / blockSize;
    spgemm_count_kernel_opt<<<gridSize, blockSize>>>(
        d_A_indptr, d_A_indices,
        d_B_indptr, d_B_indices,
        A_rows, B_cols, d_row_nnz);
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

    // Compute C = A * B
    spgemm_compute_kernel_opt<<<gridSize, blockSize>>>(
        d_A_indptr, d_A_indices, d_A_data,
        d_B_indptr, d_B_indices, d_B_data,
        d_C_indptr, d_C_indices, d_C_data,
        A_rows);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(C.indices.data(), d_C_indices, sizeof(int) * nnz_C, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C.data.data(), d_C_data, sizeof(float) * nnz_C, cudaMemcpyDeviceToHost));
    C.rows = A.rows;
    C.cols = B.cols;

    // Cleanup
    cudaFree(d_A_indptr); cudaFree(d_A_indices); cudaFree(d_A_data);
    cudaFree(d_B_indptr); cudaFree(d_B_indices); cudaFree(d_B_data);
    cudaFree(d_row_nnz);
    cudaFree(d_C_indptr); cudaFree(d_C_indices); cudaFree(d_C_data);
}
