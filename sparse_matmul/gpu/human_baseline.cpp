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

#include "spgemm_interface.h"
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <stdio.h>
#include <stdlib.h>

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

void spgemm_gpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C) {
    int *dA_offsets, *dA_columns, *dB_offsets, *dB_columns, *dC_offsets, *dC_columns;
    float *dA_values, *dB_values, *dC_values;
    int A_nnz = A.data.size();
    int B_nnz = B.data.size();
    int A_rows = A.rows;
    int A_cols = A.cols;
    int B_rows = B.rows;
    int B_cols = B.cols;

    CHECK_CUDA(cudaMalloc((void **)&dA_offsets, A.indptr.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dA_columns, A_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dA_values, A_nnz * sizeof(float)));

    CHECK_CUDA(cudaMalloc((void **)&dB_offsets, B.indptr.size() * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dB_columns, B_nnz * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&dB_values, B_nnz * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA_offsets, A.indptr.data(), A.indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_columns, A.indices.data(), A_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_values, A.data.data(), A_nnz * sizeof(float), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMemcpy(dB_offsets, B.indptr.data(), B.indptr.size() * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_columns, B.indices.data(), B_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_values, B.data.data(), B_nnz * sizeof(float), cudaMemcpyHostToDevice));

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

    std::vector<int> hC_offsets(C_num_rows + 1);
    std::vector<int> hC_columns(C_nnz);
    std::vector<float> hC_values(C_nnz);

    CHECK_CUDA(cudaMemcpy(hC_offsets.data(), dC_offsets, (C_num_rows + 1) * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_columns.data(), dC_columns, C_nnz * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_values.data(), dC_values, C_nnz * sizeof(float), cudaMemcpyDeviceToHost));

    C.indptr = std::move(hC_offsets);
    C.indices = std::move(hC_columns);
    C.data = std::move(hC_values);
    C.rows = C_num_rows;
    C.cols = C_num_cols;

    cudaFree(dA_offsets); cudaFree(dA_columns); cudaFree(dA_values);
    cudaFree(dB_offsets); cudaFree(dB_columns); cudaFree(dB_values);
    cudaFree(dC_offsets); cudaFree(dC_columns); cudaFree(dC_values);
    cudaFree(dBuffer1); cudaFree(dBuffer2);
    cusparseDestroySpMat(matA); cusparseDestroySpMat(matB); cusparseDestroySpMat(matC);
    cusparseSpGEMM_destroyDescr(spgemmDesc);
    cusparseDestroy(handle);
}
