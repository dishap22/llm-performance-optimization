#pragma once
#include "../utils/csr_io.h"

void spgemm_gpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C);