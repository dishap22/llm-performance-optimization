#include "csr_io.h"
#include <fstream>

static std::vector<int> read_vector_int(const std::string& path) {
    std::ifstream file(path);
    std::vector<int> data;
    int val;
    while (file >> val) data.push_back(val);
    return data;
}

static std::vector<float> read_vector_float(const std::string& path) {
    std::ifstream file(path);
    std::vector<float> data;
    float val;
    while (file >> val) data.push_back(val);
    return data;
}

static void write_vector_int(const std::string& path, const std::vector<int>& vec) {
    std::ofstream file(path);
    for (int v : vec) file << v << "\n";
}

static void write_vector_float(const std::string& path, const std::vector<float>& vec) {
    std::ofstream file(path);
    for (float v : vec) file << v << "\n";
}

static std::pair<int, int> read_shape(const std::string& path) {
    std::ifstream file(path);
    int rows, cols;
    file >> rows >> cols;
    return {rows, cols};
}

static void write_shape(const std::string& path, int rows, int cols) {
    std::ofstream file(path);
    file << rows << " " << cols << "\n";
}

CSRMatrix load_csr_matrix(const std::string& basename) {
    CSRMatrix mat;
    mat.indptr = read_vector_int(basename + "_indptr.txt");
    mat.indices = read_vector_int(basename + "_indices.txt");
    mat.data = read_vector_float(basename + "_data.txt");
    auto [rows, cols] = read_shape(basename + "_shape.txt");
    mat.rows = rows;
    mat.cols = cols;
    return mat;
}

void save_csr_matrix(const std::string& basename, const CSRMatrix& mat) {
    write_vector_int(basename + "_indptr.txt", mat.indptr);
    write_vector_int(basename + "_indices.txt", mat.indices);
    write_vector_float(basename + "_data.txt", mat.data);
    write_shape(basename + "_shape.txt", mat.rows, mat.cols);
}