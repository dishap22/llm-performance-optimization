#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <cstdio>
#include <unordered_map>
#include <tuple>
#include <cmath>
#include <algorithm>

extern "C" {
#include "mmio.h"   // from SuiteSparse/NIST
}

struct COOMatrix {
    int rows, cols, nnz;
    std::vector<int> row, col;
    std::vector<double> val;
};

struct CSRMatrix {
    int rows, cols;
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> data;
};

COOMatrix read_mtx_coo(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) throw std::runtime_error("Cannot open file " + path);

    MM_typecode matcode;
    if (mm_read_banner(f, &matcode) != 0) {
        throw std::runtime_error("Could not process Matrix Market banner");
    }

    int M, N, nz;
    mm_read_mtx_crd_size(f, &M, &N, &nz);

    COOMatrix A;
    A.rows = M;
    A.cols = N;

    A.row.reserve(nz * 2); // symmetric expansion may double
    A.col.reserve(nz * 2);
    A.val.reserve(nz * 2);

    for (int i = 0; i < nz; i++) {
        int r, c;
        double v = 1.0; // default for pattern

        if (mm_is_pattern(matcode)) {
            // only row and col in the file
            int ret = fscanf(f, "%d %d", &r, &c);
            if (ret != 2) throw std::runtime_error("Error reading MTX entry (pattern)");
        } else {
            int ret = fscanf(f, "%d %d %lg", &r, &c, &v);
            if (ret != 3) throw std::runtime_error("Error reading MTX entry (numeric)");
        }

        r--; c--; // 1-based → 0-based
        A.row.push_back(r);
        A.col.push_back(c);
        A.val.push_back(v);

        // expand symmetry if needed
        if ((mm_is_symmetric(matcode) || mm_is_hermitian(matcode) || mm_is_skew(matcode)) && r != c) {
            A.row.push_back(c);
            A.col.push_back(r);
            if (mm_is_skew(matcode))
                A.val.push_back(-v);
            else if (mm_is_hermitian(matcode))
                A.val.push_back(v); // real, so conj(v) = v
            else
                A.val.push_back(v);
        }
    }

    fclose(f);

    A.nnz = A.row.size();
    return A;
}

CSRMatrix coo_to_csr(const COOMatrix& coo) {
    int rows = coo.rows;
    int nnz  = coo.row.size();

    CSRMatrix A;
    A.rows = rows;
    A.cols = coo.cols;
    A.indptr.assign(rows+1, 0);
    A.indices.resize(nnz);
    A.data.resize(nnz);

    // count nonzeros per row
    for (int r : coo.row) {
        A.indptr[r+1]++;
    }

    // prefix sum
    for (int i = 0; i < rows; i++) {
        A.indptr[i+1] += A.indptr[i];
    }

    // temporary copy of indptr for insertion
    std::vector<int> counter = A.indptr;

    for (int k = 0; k < nnz; k++) {
        int r = coo.row[k];
        int dest = counter[r]++;
        A.indices[dest] = coo.col[k];
        A.data[dest]    = coo.val[k];
    }

    return A;
}

void save_csr_binary(const CSRMatrix& A, const std::string& basename) {
    std::ofstream f_indptr(basename + "_indptr.bin", std::ios::binary);
    f_indptr.write((char*)A.indptr.data(), A.indptr.size() * sizeof(int));

    std::ofstream f_indices(basename + "_indices.bin", std::ios::binary);
    f_indices.write((char*)A.indices.data(), A.indices.size() * sizeof(int));

    std::ofstream f_data(basename + "_data.bin", std::ios::binary);
    f_data.write((char*)A.data.data(), A.data.size() * sizeof(double));

    std::ofstream f_shape(basename + "_shape.bin", std::ios::binary);
    int shape[2] = {A.rows, A.cols};
    f_shape.write((char*)shape, 2 * sizeof(int));

    std::cout << "Saved CSR: " << basename << " ("
              << A.rows << " x " << A.cols
              << ", nnz=" << A.data.size() << ")\n";
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " input.mtx output_basename\n";
        return 1;
    }

    try {
        COOMatrix coo = read_mtx_coo(argv[1]);
        CSRMatrix csr = coo_to_csr(coo);
        save_csr_binary(csr, argv[2]);
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
