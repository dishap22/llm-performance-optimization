#pragma once
#include "csr_io.h"

void spgemm_cpu(const CSRMatrix& A, const CSRMatrix& B, CSRMatrix& C);