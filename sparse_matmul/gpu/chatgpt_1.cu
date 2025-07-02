// spgemm_interface.cu
#include "spgemm_interface.h"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <unordered_map>
#include <vector>
#include <numeric>
#include <cassert>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Device CSR structure
struct DeviceCSR {
    int* d_indptr;
    int* d_indices;
    float* d_data;
    int rows;
    int cols;
};

// Kernel to compute row sizes for output C matrix
// Count kernel
__global__ void spgemm_count_kernel(const int* A_indptr, const int* A_indices, const float* A_data,
                                    const int* B_indptr, const int* B_indices, const float* B_data,
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
                                      int* C_indptr, int* C_indices, float* C_data,
                                      int A_rows, int B_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= A_rows) return;

    extern __shared__ float shmem[]; // single buffer for both vals and flags

    float* vals = shmem + threadIdx.x * B_cols;
    int* flags = (int*)(shmem + blockDim.x * B_cols); // after all vals

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


void spgemm_gpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    assert(A.cols == B.rows);
    assert(B.cols < 4096);

    int A_rows = A.rows;
    int B_cols = B.cols;

    // Allocate device memory for A
    int* d_A_indptr; int* d_A_indices; float* d_A_data;
    CHECK_CUDA(cudaMalloc(&d_A_indptr, (A.rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_A_indices, A.indices.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_A_data, A.data.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_A_indptr, A.indptr.data(), (A.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_indices, A.indices.data(), A.indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_A_data, A.data.data(), A.data.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Allocate device memory for B
    int* d_B_indptr; int* d_B_indices; float* d_B_data;
    CHECK_CUDA(cudaMalloc(&d_B_indptr, (B.rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_B_indices, B.indices.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_B_data, B.data.size() * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_B_indptr, B.indptr.data(), (B.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_indices, B.indices.data(), B.indices.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B_data, B.data.data(), B.data.size() * sizeof(float), cudaMemcpyHostToDevice));

    // Step 1: Count non-zeros per row in C
    int* d_C_row_sizes;
    CHECK_CUDA(cudaMalloc(&d_C_row_sizes, A_rows * sizeof(int)));

    int blockSize = 128;
    int gridSize = (A_rows + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * B_cols * sizeof(int);
    spgemm_count_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_A_indptr, d_A_indices, d_A_data,
                                                                 d_B_indptr, d_B_indices, d_B_data,
                                                                 A_rows, B_cols, d_C_row_sizes);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy row sizes back and compute C.indptr
    std::vector<int> h_C_row_sizes(A_rows);
    CHECK_CUDA(cudaMemcpy(h_C_row_sizes.data(), d_C_row_sizes, A_rows * sizeof(int), cudaMemcpyDeviceToHost));

    C.indptr.resize(A_rows + 1, 0);
    for (int i = 0; i < A_rows; ++i) {
        C.indptr[i + 1] = C.indptr[i] + h_C_row_sizes[i];
    }

    int total_nnz = C.indptr.back();
    C.indices.resize(total_nnz);
    C.data.resize(total_nnz);

    // Copy indptr to device
    int* d_C_indptr;
    CHECK_CUDA(cudaMalloc(&d_C_indptr, (A_rows + 1) * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_C_indptr, C.indptr.data(), (A_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate C.indices and C.data on device
    int* d_C_indices;
    float* d_C_data;
    CHECK_CUDA(cudaMalloc(&d_C_indices, total_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_C_data, total_nnz * sizeof(float)));

    // Step 2: Compute actual values
    sharedMemSize = 2 * blockSize * B_cols * sizeof(float);  // for vals and flags
    spgemm_compute_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_A_indptr, d_A_indices, d_A_data,
                                                                   d_B_indptr, d_B_indices, d_B_data,
                                                                   d_C_indptr, d_C_indices, d_C_data,
                                                                   A_rows, B_cols);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(C.indices.data(), d_C_indices, total_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(C.data.data(), d_C_data, total_nnz * sizeof(float), cudaMemcpyDeviceToHost));

    C.rows = A.rows;
    C.cols = B.cols;

    // Free all device memory
    cudaFree(d_A_indptr); cudaFree(d_A_indices); cudaFree(d_A_data);
    cudaFree(d_B_indptr); cudaFree(d_B_indices); cudaFree(d_B_data);
    cudaFree(d_C_row_sizes); cudaFree(d_C_indptr); cudaFree(d_C_indices); cudaFree(d_C_data);
}
