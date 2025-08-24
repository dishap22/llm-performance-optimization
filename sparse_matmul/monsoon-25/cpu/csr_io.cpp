#include "csr_io.h"
#include <fstream>
#include <stdexcept>

// Generic helper function to read a binary file into a vector
template<typename T>
static std::vector<T> read_vector_binary(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<T> buffer(size / sizeof(T));
    if (!file.read((char*)buffer.data(), size)) {
        throw std::runtime_error("Error reading file: " + path);
    }
    return buffer;
}

// Generic helper function to write a vector to a binary file
template<typename T>
static void write_vector_binary(const std::string& path, const std::vector<T>& vec) {
    std::ofstream file(path, std::ios::binary);
     if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + path);
    }
    file.write((char*)vec.data(), vec.size() * sizeof(T));
}


// --- Implementation of the new binary loading function ---
CSRMatrix load_csr_matrix_binary(const std::string& basename) {
    CSRMatrix mat;

    // Read shape
    std::ifstream shape_file(basename + "_shape.bin", std::ios::binary);
     if (!shape_file.is_open()) {
        throw std::runtime_error("Cannot open file: " + basename + "_shape.bin");
    }
    int shape[2];
    shape_file.read((char*)shape, 2 * sizeof(int));
    mat.rows = shape[0];
    mat.cols = shape[1];
    shape_file.close();

    // Read CSR data
    mat.indptr = read_vector_binary<int>(basename + "_indptr.bin");
    mat.indices = read_vector_binary<int>(basename + "_indices.bin");
    mat.data = read_vector_binary<double>(basename + "_data.bin"); // Reading as double

    return mat;
}

// --- Implementation of the new binary saving function ---
void save_csr_matrix_binary(const std::string& basename, const CSRMatrix& mat) {
    // Save shape
    std::ofstream shape_file(basename + "_shape.bin", std::ios::binary);
    int shape[2] = {mat.rows, mat.cols};
    shape_file.write((char*)shape, 2 * sizeof(int));
    shape_file.close();

    // Save CSR data
    write_vector_binary(basename + "_indptr.bin", mat.indptr);
    write_vector_binary(basename + "_indices.bin", mat.indices);
    write_vector_binary(basename + "_data.bin", mat.data);
}

static std::vector<int> read_vector_int(const std::string& path) {
    std::ifstream file(path);
    std::vector<int> data;
    int val;
    while (file >> val) data.push_back(val);
    return data;
}

static std::vector<double> read_vector_double(const std::string& path) {
    std::ifstream file(path);
    std::vector<double> data;
    double val;
    while (file >> val) data.push_back(val);
    return data;
}

static void write_vector_int(const std::string& path, const std::vector<int>& vec) {
    std::ofstream file(path);
    for (int v : vec) file << v << "\n";
}

static void write_vector_double(const std::string& path, const std::vector<double>& vec) {
    std::ofstream file(path);
    for (double v : vec) file << v << "\n";
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
    mat.data = read_vector_double(basename + "_data.txt");
    auto [rows, cols] = read_shape(basename + "_shape.txt");
    mat.rows = rows;
    mat.cols = cols;
    return mat;
}

void save_csr_matrix(const std::string& basename, const CSRMatrix& mat) {
    write_vector_int(basename + "_indptr.txt", mat.indptr);
    write_vector_int(basename + "_indices.txt", mat.indices);
    write_vector_double(basename + "_data.txt", mat.data);
    write_shape(basename + "_shape.txt", mat.rows, mat.cols);
}