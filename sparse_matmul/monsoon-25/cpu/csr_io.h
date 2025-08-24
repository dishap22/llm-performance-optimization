#pragma once
#include <vector>
#include <string>

struct CSRMatrix {
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> data;
    int rows;
    int cols;
};

CSRMatrix load_csr_matrix(const std::string& basename);
void save_csr_matrix(const std::string& basename, const CSRMatrix& mat);

CSRMatrix load_csr_matrix_binary(const std::string& basename);
void save_csr_matrix_binary(const std::string& basename, const CSRMatrix& mat);