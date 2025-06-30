/*
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Portions of this code are adapted from the NVIDIA CUDA Library Samples:
 * https://github.com/NVIDIA/CUDALibrarySamples/tree/main/cuSPARSE/spgemm
 *
 * Copyright (c) 2022-2024 NVIDIA CORPORATION AND AFFILIATES.
 *
 * This source code is licensed under the BSD 3-Clause License found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("cuSPARSE API failed at line %d with error: %d\n",              \
               __LINE__, status);                                              \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
}

template <typename T>
std::vector<T> read_vector_txt(const std::string& path) {
    std::ifstream file(path);
    std::vector<T> data;
    T val;
    while (file >> val) {
        data.push_back(val);
    }
    return data;
}

std::pair<int, int> read_matrix_shape(const std::string& path) {
    std::ifstream file(path);
    int rows, cols;
    file >> rows >> cols;
    return {rows, cols};
}

void write_vector_txt(const std::string& path, const std::vector<int>& vec) {
    std::ofstream file(path);
    for (int v : vec) file << v << "\n";
}

void write_vector_txt(const std::string& path, const std::vector<float>& vec) {
    std::ofstream file(path);
    for (float v : vec) file << v << "\n";
}

int main() {
    const std::string input_dir = "../data";

    // Load A
    std::vector<int> hA_offsets  = read_vector_txt<int>(input_dir + "/py_A_indptr.txt");
    std::vector<int> hA_columns  = read_vector_txt<int>(input_dir + "/py_A_indices.txt");
    std::vector<float> hA_values = read_vector_txt<float>(input_dir + "/py_A_data.txt");
    auto [A_rows, A_cols] = read_matrix_shape(input_dir + "/py_A_shape.txt");
    int A_nnz = hA_values.size();

    // Load B
    std::vector<int> hB_offsets  = read_vector_txt<int>(input_dir + "/py_B_indptr.txt");
    std::vector<int> hB_columns  = read_vector_txt<int>(input_dir + "/py_B_indices.txt");
    std::vector<float> hB_values = read_vector_txt<float>(input_dir + "/py_B_data.txt");
    auto [B_rows, B_cols] = read_matrix_shape(input_dir + "/py_B_shape.txt");
    int B_nnz = hB_values.size();

    // Allocate device memory
    int *dA_offsets, *dA_columns, *dB_offsets, *dB_columns, *dC_offsets, *dC_columns;
    float *dA_values, *dB_values, *dC_values;

    CHECK_CUDA(cudaMalloc((void **)&dA_offsets, hA_offsets.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dA_values, A_nnz * sizeof(float)));

    CHECK_CUDA(cudaMalloc((void **)&dB_offsets, hB_offsets.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dB_columns, B_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dB_values, B_nnz * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA_offsets, hA_offsets.data(), hA_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_columns, hA_columns.data(), A_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, hA_values.data(), A_nnz * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(dB_offsets, hB_offsets.data(), hB_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_columns, hB_columns.data(), B_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_values, hB_values.data(), B_nnz * sizeof(float), cudaMemcpyHostToDevice));

    // cuSPARSE setup
    cusparseHandle_t handle;
    CHECK_CUSPARSE(cusparseCreate(&handle));

    cusparseSpMatDescr_t matA, matB, matC;
    CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_rows, A_cols, A_nnz,
        dA_offsets, dA_columns, dA_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CHECK_CUSPARSE(cusparseCreateCsr(&matB, B_rows, B_cols, B_nnz,
        dB_offsets, dB_columns, dB_values, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    CHECK_CUDA(cudaMalloc((void **)&dC_offsets, (A_rows + 1) * sizeof(int)));
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, A_rows, B_cols, 0,
        dC_offsets, NULL, NULL, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    // SpGEMM
    float alpha = 1.0f, beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;
    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc));

    size_t bufferSize1, bufferSize2;
    void *dBuffer1 = NULL, *dBuffer2 = NULL;

    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha,
        matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize1, NULL));
    CHECK_CUDA(cudaMalloc((void **)&dBuffer1, bufferSize1));
    CHECK_CUSPARSE(cusparseSpGEMM_workEstimation(handle, opA, opB, &alpha,
        matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize1, dBuffer1));

    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB, &alpha,
        matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize2, NULL));
    CHECK_CUDA(cudaMalloc((void **)&dBuffer2, bufferSize2));
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB, &alpha,
        matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc, &bufferSize2, dBuffer2));

    int64_t C_num_rows, C_num_cols, C_nnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &C_num_rows, &C_num_cols, &C_nnz));

    CHECK_CUDA(cudaMalloc((void **)&dC_columns, C_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dC_values,  C_nnz * sizeof(float)));
    CHECK_CUSPARSE(cusparseCsrSetPointers(matC, dC_offsets, dC_columns, dC_values));

    CHECK_CUSPARSE(cusparseSpGEMM_copy(handle, opA, opB, &alpha,
        matA, matB, &beta, matC, computeType, CUSPARSE_SPGEMM_DEFAULT,
        spgemmDesc));

    // Copy C to host
    std::vector<int> hC_offsets(A_rows + 1);
    std::vector<int> hC_columns(C_nnz);
    std::vector<float> hC_values(C_nnz);

    CHECK_CUDA(cudaMemcpy(hC_offsets.data(), dC_offsets, (A_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_columns.data(), dC_columns, C_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_values.data(), dC_values, C_nnz * sizeof(float), cudaMemcpyDeviceToHost));

    // Save C
    write_vector_txt("C_indptr.txt", hC_offsets);
    write_vector_txt("C_indices.txt", hC_columns);
    write_vector_txt("C_data.txt", hC_values);
    std::cout << "Saved C matrix to C_indptr.txt, C_indices.txt, C_data.txt\n";

    // Cleanup
    cudaFree(dA_offsets); cudaFree(dA_columns); cudaFree(dA_values);
    cudaFree(dB_offsets); cudaFree(dB_columns); cudaFree(dB_values);
    cudaFree(dC_offsets); cudaFree(dC_columns); cudaFree(dC_values);
    cudaFree(dBuffer1); cudaFree(dBuffer2);
    cusparseDestroySpMat(matA); cusparseDestroySpMat(matB); cusparseDestroySpMat(matC);
    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroy(handle);
    return 0;
}
