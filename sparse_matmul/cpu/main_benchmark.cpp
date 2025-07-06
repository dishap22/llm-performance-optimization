#include "../utils/csr_io.h"
#include "spgemm_interface.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <thread>
#include <cstring>

void print_system_info() {
    std::cout << "=== CPU System Information ===" << std::endl;
    std::cout << "Number of CPU cores: " << std::thread::hardware_concurrency() << std::endl;

    // Try to get more detailed CPU info on Linux
    #ifdef __linux__
    FILE* file = fopen("/proc/cpuinfo", "r");
    if (file) {
        char line[256];
        while (fgets(line, sizeof(line), file)) {
            if (strncmp(line, "model name", 10) == 0) {
                char* model = strchr(line, ':');
                if (model) {
                    model++; // Skip the colon
                    while (*model == ' ' || *model == '\t') model++; // Skip whitespace
                    std::cout << "CPU Model: " << model;
                    break;
                }
            }
        }
        fclose(file);
    }
    #endif

    std::cout << "===============================" << std::endl;
}

int main() {
    try {
        // Print system information
        print_system_info();

        // Load matrices
        const std::string input_dir = "../data/py_";
        std::cout << "Loading matrices from " << input_dir << "..." << std::endl;

        CSRMatrix A = load_csr_matrix(input_dir + "A");
        CSRMatrix B = load_csr_matrix(input_dir + "B");

        std::cout << "Matrix A: " << A.rows << "x" << A.cols << ", " << A.data.size() << " nonzeros" << std::endl;
        std::cout << "Matrix B: " << B.rows << "x" << B.cols << ", " << B.data.size() << " nonzeros" << std::endl;

        // Calculate expected operations for performance measurement
        long long total_ops = 0;
        for (int i = 0; i < A.rows; ++i) {
            for (int jj = A.indptr[i]; jj < A.indptr[i+1]; ++jj) {
                int a_col = A.indices[jj];
                total_ops += B.indptr[a_col+1] - B.indptr[a_col];
            }
        }
        long long total_flops = 2 * total_ops; // Each multiply-add is 2 FLOPS

        std::cout << "Expected operations: " << total_ops << " multiplies (" << total_flops << " FLOPS)" << std::endl;
        std::cout << std::endl;

        // Result matrix
        CSRMatrix C;

        // Warmup run
        std::cout << "Running warmup..." << std::endl;
        spgemm_cpu(A, B, C);

        // Benchmark runs
        const int num_runs = 10;
        std::vector<double> times;
        times.reserve(num_runs);

        std::cout << "Running " << num_runs << " benchmark iterations..." << std::endl;

        for (int run = 0; run < num_runs; ++run) {
            // Clear result matrix
            C.indptr.clear();
            C.indices.clear();
            C.data.clear();

            // Start timing
            auto start = std::chrono::high_resolution_clock::now();

            // Run SpGEMM
            spgemm_cpu(A, B, C);

            // End timing
            auto end = std::chrono::high_resolution_clock::now();

            // Calculate elapsed time
            std::chrono::duration<double> elapsed = end - start;
            times.push_back(elapsed.count());

            // Print progress
            std::cout << "  Run " << (run + 1) << "/" << num_runs
                      << ": " << std::fixed << std::setprecision(4) << elapsed.count() << " s" << std::endl;
        }

        // Calculate statistics
        double mean_time = 0.0, std_time = 0.0;
        double min_time = times[0], max_time = times[0];

        for (double t : times) {
            mean_time += t;
            min_time = std::min(min_time, t);
            max_time = std::max(max_time, t);
        }
        mean_time /= times.size();

        for (double t : times) {
            std_time += (t - mean_time) * (t - mean_time);
        }
        std_time = std::sqrt(std_time / (times.size() - 1));

        // Calculate performance
        double gflops = total_flops / (mean_time * 1e9);

        // Print results in SGEMM_CUDA format
        std::cout << std::endl;
        std::cout << "=== Benchmark Results ===" << std::endl;
        std::cout << "Matrix size: " << A.rows << "x" << A.rows << std::endl;
        std::cout << "Matrix density: " << std::fixed << std::setprecision(3)
                  << (double)(A.data.size()) / (A.rows * A.cols) << std::endl;
        std::cout << "Operations: " << total_ops << " multiplies (" << total_flops << " FLOPS)" << std::endl;
        std::cout << "Result matrix: " << C.rows << "x" << C.cols << ", " << C.data.size() << " nonzeros" << std::endl;
        std::cout << std::endl;

        std::cout << "Timing Statistics:" << std::endl;
        std::cout << "  Mean time: " << std::fixed << std::setprecision(6) << mean_time << " s" << std::endl;
        std::cout << "  Std dev:   " << std::fixed << std::setprecision(6) << std_time << " s" << std::endl;
        std::cout << "  Min time:  " << std::fixed << std::setprecision(6) << min_time << " s" << std::endl;
        std::cout << "  Max time:  " << std::fixed << std::setprecision(6) << max_time << " s" << std::endl;
        std::cout << std::endl;

        std::cout << "Performance:" << std::endl;
        std::cout << "  GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
        std::cout << std::endl;

        // Output in SGEMM_CUDA format for parsing
        std::cout << "Time: " << std::fixed << std::setprecision(6) << mean_time << " s" << std::endl;

        // Save result matrix
        save_csr_matrix("C", C);
        std::cout << "Result matrix saved to C_*.txt files" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown error occurred" << std::endl;
        return 1;
    }
}