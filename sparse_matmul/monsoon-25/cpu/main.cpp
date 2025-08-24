#include "csr_io.h"
#include "spgemm_interface.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>
#include <omp.h>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <matrix_name>\n";
        return 1;
    }

    // --- Configuration for benchmarking ---
    const int num_runs = 10;

    std::string matrix_name = argv[1];
    std::string input_path = "../data/" + matrix_name + "/" + matrix_name;

    try {
        std::cout << "Loading matrix: " << matrix_name << " from " << input_path << std::endl;
        CSRMatrix A = load_csr_matrix_binary(input_path);
        std::cout << "Matrix A loaded: " << A.rows << "x" << A.cols << ", nnz=" << A.data.size() << std::endl;

        CSRMatrix C;

        // --- 1. Warm-up Run (optional but recommended) ---
        std::cout << "Performing one warm-up run..." << std::endl;
        spgemm_cpu(A, A, C); // The result of this is discarded, it just warms the cache.

        // --- 2. Timed Runs ---
        std::cout << "Computing SpGEMM, averaging over " << num_runs << " runs..." << std::endl;
        double total_execution_time = 0.0;

        for (int i = 0; i < num_runs; ++i) {
            double start_time = omp_get_wtime();
            spgemm_cpu(A, A, C);
            double end_time = omp_get_wtime();
            total_execution_time += (end_time - start_time);
        }

        // --- 3. Calculate and Print Average ---
        double average_time = total_execution_time / num_runs;

        long long nnz_C = C.data.size();
        double total_flops = 2.0 * nnz_C * num_runs;
        double gflops = (total_flops / total_execution_time) / 1e9;

        std::cout << "Average Execution Time (" << num_runs << " runs): " << average_time << " seconds" << std::endl;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

        std::cout << "SpGEMM complete. Result matrix C: " << C.rows << "x" << C.cols << ", nnz=" << C.data.size() << std::endl;

        std::string output_basename = "C_openmp_" + matrix_name;
        save_csr_matrix_binary(output_basename, C);
        std::cout << "Result saved to " << output_basename << "_*.bin" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}