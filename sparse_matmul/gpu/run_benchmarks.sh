#!/usr/bin/env bash

set -euo pipefail

# This script runs the SpGEMM benchmarking suite for all implementations
# and generates performance plots and reports

echo "=== SpGEMM GPU Benchmarking Suite ==="
echo "Starting comprehensive performance evaluation..."
echo

# Create benchmark results directory
mkdir -p benchmark_results

# Check if required files exist
required_files=("human_baseline.cpp" "chatgpt_5.cu" "gemini_1.cu")
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "Warning: $file not found, some benchmarks may be skipped"
    fi
done

# Run the main benchmarking script
echo "Running benchmark_spgemm.py..."
python3 benchmark_spgemm.py

# Check if benchmark completed successfully
if [[ $? -eq 0 ]]; then
    echo
    echo "=== Benchmarking completed successfully! ==="
    echo "Results saved to:"
    echo "  - benchmark_results.csv (detailed results)"
    echo "  - benchmark_results.png (performance plots)"
    echo
    echo "You can now analyze the results or run additional analysis."
else
    echo
    echo "=== Benchmarking failed! ==="
    echo "Check the error messages above for details."
    exit 1
fi