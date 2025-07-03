#include "../utils/csr_io.h"
#include "spgemm_interface.h"
#include <iostream>

int main() {
    const std::string input_dir = "../data/py_";
    CSRMatrix A = load_csr_matrix(input_dir + "A");
    CSRMatrix B = load_csr_matrix(input_dir + "B");
    CSRMatrix C;
    spgemm_cpu(A, B, C);
    save_csr_matrix("C", C);
    std::cout << "Saved C matrix to C_indptr.txt, C_indices.txt, C_data.txt, C_shape.txt\n";
    return 0;
}