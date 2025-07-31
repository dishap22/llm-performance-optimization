#!/usr/bin/env python3

import subprocess
import numpy as np
import matplotlib.pyplot as plt
import re
import os
import time
import pandas as pd
from pathlib import Path

# Settings
matrix_sizes = [256, 512, 1024, 2048]  # Larger sizes for better GPU utilization
densities = [0.01, 0.05, 0.1]     # Different sparsity patterns
warmup_runs = 3                         # Number of warmup runs
benchmark_runs = 5                     # Number of benchmark runs
gpu_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(gpu_dir, '../utils'))
exec_name = "spgemm_benchmark"
main_cpp = "main_benchmark.cpp"
csr_io_cpp = "../utils/csr_io.cpp"

# Implementation mapping (similar to SGEMM_CUDA kernel names)
IMPLEMENTATIONS = {
    "human_baseline.cpp": "cuSPARSE Baseline",
    "chatgpt_5.cu": "ChatGPT 1st Implementation",
    "gemini_1.cu": "Gemini 1st Implementation",
    "chatgpt_7.cu": "ChatGPT 2nd Implementation",
    "gemini_7.cu": "Gemini 2nd Implementation",
    "chatgpt_10.cu": "ChatGPT 3rd Implementation",
    "gemini_14.cu": "Gemini 3rd Implementation"
}

def generate_matrices(n, density=0.1):
    """Generate test matrices using the existing CSR generator"""
    subprocess.run([
        "python3", os.path.join(utils_dir, "csr_generator.py"),
        "--rows", str(n), "--cols", str(n), "--density", str(density)
    ], check=True, capture_output=True)

def compile_impl(impl):
    """Compile a specific implementation"""
    # Remove old exec
    exec_path = os.path.join(gpu_dir, exec_name)
    if os.path.exists(exec_path):
        os.remove(exec_path)

    # Compile
    src = os.path.join(gpu_dir, impl)
    main = os.path.join(gpu_dir, main_cpp)
    csr_io = os.path.join(gpu_dir, "../utils/csr_io.cpp")

    # Use nvcc for .cu files and human_baseline.cpp (which contains CUDA code)
    if impl.endswith(".cu") or impl == "human_baseline.cpp":
        cmd = [
            "nvcc", "-std=c++17", "-I../utils", main, csr_io, src,
            "-o", exec_path, "-O3"  # Add optimization flags
        ]
        # Add cuSPARSE for human_baseline
        if "human_baseline" in impl:
            cmd.append("-lcusparse")
    else:
        # For other .cpp files, use g++
        cmd = [
            "g++", "-std=c++17", "-I../utils", main, csr_io, src,
            "-o", exec_path, "-O3"
        ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return exec_path
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed for {impl}: {e}")
        print(f"stdout: {e.stdout.decode()}")
        print(f"stderr: {e.stderr.decode()}")
        return None

def run_spgemm_benchmark(exec_path, warmup_runs=3, benchmark_runs=10):
    """Run SpGEMM benchmark with warmup and multiple runs"""
    times = []

    try:
        # Warmup runs
        for _ in range(warmup_runs):
            result = subprocess.run([exec_path], capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"  Warmup run failed: {result.stderr}")
                return None

        # Benchmark runs
        for run in range(benchmark_runs):
            result = subprocess.run([exec_path], capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"  Benchmark run {run} failed: {result.stderr}")
                return None

            # Parse timing from output
            match = re.search(r"Time:\s*([0-9.]+)\s*s", result.stdout)
            if match:
                times.append(float(match.group(1)))
            else:
                # Check for CUDA errors
                if "out of memory" in result.stderr.lower() or "out of memory" in result.stdout.lower():
                    print(f"  CUDA out of memory error")
                    return None
                elif "CUDA API failed" in result.stderr or "CUDA API failed" in result.stdout:
                    print(f"  CUDA API error")
                    return None
                else:
                    print(f"  Run {run} - Output: {result.stdout}")
                    print(f"  Run {run} - Error: {result.stderr}")
                    return None

        # Calculate statistics
        if len(times) == benchmark_runs:
            return {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'times': times
            }
        else:
            return None

    except subprocess.TimeoutExpired:
        print(f"  Timeout after 5 minutes")
        return None
    except Exception as e:
        print(f"  Runtime error: {e}")
        return None

def count_operations(A_indptr, A_indices, B_indptr):
    """Count the number of floating point operations"""
    total = 0
    for i in range(len(A_indptr) - 1):
        for jj in range(A_indptr[i], A_indptr[i+1]):
            a_col = A_indices[jj]
            total += B_indptr[a_col+1] - B_indptr[a_col]
    return total

def load_matrix_stats(path_prefix):
    """Load matrix statistics for performance calculation"""
    indptr = np.loadtxt(f"../data/py_{path_prefix}_indptr.txt", dtype=int)
    indices = np.loadtxt(f"../data/py_{path_prefix}_indices.txt", dtype=int)
    data = np.loadtxt(f"../data/py_{path_prefix}_data.txt", dtype=float)
    nnz = len(data)
    rows = len(indptr) - 1
    cols = max(indices) + 1 if len(indices) > 0 else 1
    return nnz, rows, cols

def load_indptr_indices(path_prefix):
    """Load indptr and indices for operation counting"""
    indptr = np.loadtxt(f"../data/py_{path_prefix}_indptr.txt", dtype=int)
    indices = np.loadtxt(f"../data/py_{path_prefix}_indices.txt", dtype=int)
    return indptr, indices

def plot_results(results_df, save_path="benchmark_results.png"):
    """Create performance plots similar to SGEMM_CUDA"""
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['font.family'] = 'monospace'

    # Check if we have any results
    if results_df.empty:
        print("No benchmark results available for plotting")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Performance vs Matrix Size (fixed density)
    density = 0.1
    df_size = results_df[results_df['density'] == density]

    for impl in df_size['implementation'].unique():
        impl_data = df_size[df_size['implementation'] == impl]
        ax1.plot(impl_data['size'], impl_data['gflops'], marker='o',
                label=IMPLEMENTATIONS.get(impl, impl), linewidth=2, markersize=6)

    ax1.set_xlabel('Matrix Size (N x N)')
    ax1.set_ylabel('GFLOPS')
    ax1.set_title(f'SpGEMM Performance vs Matrix Size (Density: {density})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Plot 2: Performance vs Density (fixed size)
    size = 1024
    df_density = results_df[results_df['size'] == size]

    for impl in df_density['implementation'].unique():
        impl_data = df_density[df_density['implementation'] == impl]
        ax2.plot(impl_data['density'], impl_data['gflops'], marker='s',
                label=IMPLEMENTATIONS.get(impl, impl), linewidth=2, markersize=6)

    ax2.set_xlabel('Matrix Density')
    ax2.set_ylabel('GFLOPS')
    ax2.set_title(f'SpGEMM Performance vs Density (Size: {size}x{size})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Plot 3: Relative Performance (normalized to cuSPARSE)
    baseline = "human_baseline.cpp"
    df_rel = results_df.copy()

    # Calculate relative performance
    for (size, density), group in df_rel.groupby(['size', 'density']):
        baseline_group = group[group['implementation'] == baseline]
        if not baseline_group.empty:
            baseline_perf = baseline_group['gflops'].iloc[0]
            if baseline_perf > 0:
                df_rel.loc[group.index, 'relative_perf'] = group['gflops'] / baseline_perf

        # Plot relative performance for size=1024, density=0.1
    df_rel_plot = df_rel[(df_rel['size'] == 1024) & (df_rel['density'] == 0.1)]

    if not df_rel_plot.empty:
        implementations = [impl for impl in df_rel_plot['implementation'].unique() if impl != baseline]
        if implementations:
            rel_perfs = []
            impl_names = []
            for impl in implementations:
                impl_data = df_rel_plot[df_rel_plot['implementation'] == impl]
                if not impl_data.empty and 'relative_perf' in impl_data.columns:
                    rel_perf = impl_data['relative_perf'].iloc[0]
                    if not pd.isna(rel_perf):
                        rel_perfs.append(rel_perf)
                        impl_names.append(IMPLEMENTATIONS.get(impl, impl))

            if rel_perfs:
                bars = ax3.bar(impl_names, rel_perfs, color=['#ff7f0e', '#2ca02c'][:len(rel_perfs)])
                ax3.set_ylabel('Relative Performance (cuSPARSE = 1.0)')
                ax3.set_title('Relative Performance Comparison')
                ax3.grid(True, alpha=0.3, axis='y')

                # Add value labels on bars
                for bar, perf in zip(bars, rel_perfs):
                    height = bar.get_height()
                    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f'{perf:.2f}x', ha='center', va='bottom')
            else:
                ax3.text(0.5, 0.5, 'No relative performance data available',
                        ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Relative Performance Comparison')
        else:
            ax3.text(0.5, 0.5, 'No comparison data available',
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Relative Performance Comparison')
    else:
        ax3.text(0.5, 0.5, 'No data for size=1024, density=0.1',
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Relative Performance Comparison')

    # Plot 4: Memory efficiency (GFLOPS per GB of memory)
    # This is a simplified calculation - in practice you'd measure actual memory usage
    df_mem = results_df.copy()
    df_mem['memory_gb'] = df_mem['size'] * df_mem['size'] * 4 * df_mem['density'] * 2 / 1e9  # Rough estimate
    df_mem['gflops_per_gb'] = df_mem['gflops'] / df_mem['memory_gb']

    df_mem_plot = df_mem[df_mem['size'] == 1024]

    for impl in df_mem_plot['implementation'].unique():
        impl_data = df_mem_plot[df_mem_plot['implementation'] == impl]
        ax4.plot(impl_data['density'], impl_data['gflops_per_gb'], marker='^',
                label=IMPLEMENTATIONS.get(impl, impl), linewidth=2, markersize=6)

    ax4.set_xlabel('Matrix Density')
    ax4.set_ylabel('GFLOPS/GB')
    ax4.set_title('Memory Efficiency (Size: 1024x1024)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    ax4.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main benchmarking function"""
    print("=== SpGEMM GPU Benchmarking Suite ===")
    print(f"Matrix sizes: {matrix_sizes}")
    print(f"Densities: {densities}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Benchmark runs: {benchmark_runs}")
    print()

    # Verify implementations exist
    valid_implementations = []
    for impl in IMPLEMENTATIONS.keys():
        if os.path.exists(os.path.join(gpu_dir, impl)):
            valid_implementations.append(impl)
        else:
            print(f"Warning: {impl} not found, skipping...")

    if not valid_implementations:
        print("No valid implementations found!")
        return

    print(f"Benchmarking: {[IMPLEMENTATIONS[impl] for impl in valid_implementations]}")
    print()

    # Results storage
    results = []

    # Benchmark loop
    for size in matrix_sizes:
        for density in densities:
            print(f"=== Matrix: {size}x{size}, Density: {density} ===")

            # Generate test matrices
            generate_matrices(size, density)

            # Load matrix statistics
            A_nnz, A_rows, A_cols = load_matrix_stats("A")
            B_nnz, B_rows, B_cols = load_matrix_stats("B")
            print(f"Matrix A: {A_rows}x{A_cols}, {A_nnz} nonzeros")
            print(f"Matrix B: {B_rows}x{B_cols}, {B_nnz} nonzeros")

            # Count operations
            A_indptr, A_indices = load_indptr_indices("A")
            B_indptr, _ = load_indptr_indices("B")
            num_multiplies = count_operations(A_indptr, A_indices, B_indptr)
            total_flops = 2 * num_multiplies
            print(f"Operations: {num_multiplies} multiplies ({total_flops} FLOPS)")

            for impl in valid_implementations:
                print(f"  Testing {IMPLEMENTATIONS[impl]}...")

                # Compile implementation
                exec_path = compile_impl(impl)
                if exec_path is None:
                    print(f"    Compilation failed, skipping...")
                    continue

                # Run benchmark
                benchmark_result = run_spgemm_benchmark(exec_path, warmup_runs, benchmark_runs)

                if benchmark_result is not None:
                    gflops = total_flops / (benchmark_result['mean'] * 1e9)
                    results.append({
                        'implementation': impl,
                        'size': size,
                        'density': density,
                        'time_mean': benchmark_result['mean'],
                        'time_std': benchmark_result['std'],
                        'time_min': benchmark_result['min'],
                        'time_max': benchmark_result['max'],
                        'gflops': gflops,
                        'operations': num_multiplies,
                        'total_flops': total_flops
                    })
                    print(f"    {gflops:.2f} GFLOPS ({benchmark_result['mean']:.4f} ± {benchmark_result['std']:.4f} s)")
                else:
                    print(f"    Benchmark failed")

            print()

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save results
    results_df.to_csv('benchmark_results.csv', index=False)
    print("Results saved to benchmark_results.csv")

    # Create plots
    plot_results(results_df)

    # Print summary table
    print("\n=== Performance Summary ===")
    summary = results_df.groupby('implementation')['gflops'].agg(['mean', 'std', 'min', 'max']).round(2)
    print(summary)

    # Print relative performance
    print("\n=== Relative Performance (vs cuSPARSE) ===")
    baseline_perf = results_df[results_df['implementation'] == 'human_baseline.cpp']['gflops'].mean()
    for impl in valid_implementations:
        if impl != 'human_baseline.cpp':
            impl_perf = results_df[results_df['implementation'] == impl]['gflops'].mean()
            rel_perf = impl_perf / baseline_perf
            print(f"{IMPLEMENTATIONS[impl]}: {rel_perf:.2f}x")

if __name__ == "__main__":
    main()