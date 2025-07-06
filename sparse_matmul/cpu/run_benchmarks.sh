#!/bin/bash

# CPU SpGEMM Benchmark Runner
# This script runs the CPU benchmarking for all implementations

set -e  # Exit on any error

echo "=== CPU SpGEMM Benchmarking ==="
echo "Starting CPU benchmarks..."
echo

# Check if we're in the right directory
if [ ! -f "benchmark_spgemm.py" ]; then
    echo "Error: benchmark_spgemm.py not found. Please run this script from the cpu directory."
    exit 1
fi

# Check if required tools are available
if ! command -v g++ &> /dev/null; then
    echo "Error: g++ compiler not found. Please install g++."
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install python3."
    exit 1
fi

# Check for Intel MKL (optional, will skip human_baseline if not available)
if [ ! -d "/opt/intel/oneapi/mkl/latest" ]; then
    echo "Warning: Intel MKL not found at /opt/intel/oneapi/mkl/latest"
    echo "Intel MKL baseline may not compile. Install Intel oneAPI for MKL support."
    echo
fi

# Create data directory if it doesn't exist
mkdir -p ../data

# Run the benchmark
echo "Running CPU SpGEMM benchmarks..."
python3 benchmark_spgemm.py

echo
echo "=== CPU Benchmarking Complete ==="
echo "Results saved to:"
echo "  - benchmark_results.csv"
echo "  - benchmark_results.png"
echo "  - benchmark_results/ (detailed logs)"