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
matrix_sizes = [256, 512, 1024, 2048, 4096]  # Matrix sizes to test
densities = [0.05, 0.1, 0.2]     # Different sparsity patterns
warmup_runs = 3                         # Number of warmup runs
benchmark_runs = 5                     # Number of benchmark runs
cpu_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.abspath(os.path.join(cpu_dir, '../utils'))
exec_name = "spgemm_cpu_benchmark"
main_cpp = "main_benchmark.cpp"
csr_io_cpp = "../utils/csr_io.cpp"

# Implementation mapping
IMPLEMENTATIONS = {
    "human_baseline.cpp": "Intel MKL Baseline",
    "chatgpt_2.cpp": "ChatGPT 1st Implementation",
    "gemini_1.cpp": "Gemini 1st Implementation",
    "chatgpt_3.cpp": "ChatGPT 2nd Implementation",
    "gemini_2.cpp": "Gemini 2nd Implementation",
    "chatgpt_4.cpp": "ChatGPT 3rd Implementation",
    "gemini_3.cpp": "Gemini 3rd Implementation"
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
    exec_path = os.path.join(cpu_dir, exec_name)
    if os.path.exists(exec_path):
        os.remove(exec_path)

    # Compile
    src = os.path.join(cpu_dir, impl)
    main = os.path.join(cpu_dir, main_cpp)
    csr_io = os.path.join(cpu_dir, "../utils/csr_io.cpp")

        # Use g++ for all CPU implementations
    cmd = [
        "g++", "-std=c++17", "-I../utils", main, csr_io, src,
        "-o", exec_path, "-O3", "-fopenmp"  # Add OpenMP support
    ]

    # Add MKL for human_baseline (using the path from README)
    if "human_baseline" in impl:
        # Source Intel oneAPI environment and compile
        cmd = [
            "bash", "-c",
            f"source /opt/intel/oneapi/setvars.sh && "
            f"g++ -std=c++17 -I../utils -I/opt/intel/oneapi/mkl/latest/include "
            f"{main} {csr_io} {src} -L/opt/intel/oneapi/mkl/latest/lib/intel64 "
            f"-lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lmkl_core -lpthread -lm -ldl "
            f"-o {exec_path} -O3"
        ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return exec_path
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed for {impl}: {e}")
        print(f"stdout: {e.stdout.decode()}")
        print(f"stderr: {e.stderr.decode()}")
        return None

def run_spgemm_benchmark(exec_path, warmup_runs=3, benchmark_runs=10, use_mkl=False):
    """Run SpGEMM benchmark with warmup and multiple runs"""
    times = []

    try:
        # Warmup runs
        for _ in range(warmup_runs):
            if use_mkl:
                result = subprocess.run(["bash", "-c", f"source /opt/intel/oneapi/setvars.sh && {exec_path}"],
                                      capture_output=True, text=True, timeout=300)
            else:
                result = subprocess.run([exec_path], capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"  Warmup run failed: {result.stderr}")
                return None

        # Benchmark runs
        for run in range(benchmark_runs):
            if use_mkl:
                result = subprocess.run(["bash", "-c", f"source /opt/intel/oneapi/setvars.sh && {exec_path}"],
                                      capture_output=True, text=True, timeout=300)
            else:
                result = subprocess.run([exec_path], capture_output=True, text=True, timeout=300)

            if result.returncode != 0:
                print(f"  Benchmark run {run} failed: {result.stderr}")
                return None

            # Parse timing from output
            match = re.search(r"Time:\s*([0-9.]+)\s*s", result.stdout)
            if match:
                times.append(float(match.group(1)))
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
    ax1.set_title(f'CPU SpGEMM Performance vs Matrix Size (Density: {density})')
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
    ax2.set_title(f'CPU SpGEMM Performance vs Density (Size: {size}x{size})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Plot 3: Relative Performance (normalized to MKL)
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
                ax3.set_ylabel('Relative Performance (MKL = 1.0)')
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
    print(f"Results plotted and saved to {save_path}")

def main():
    """Main benchmarking function"""
    print("=== CPU SpGEMM Benchmarking ===")
    print(f"Matrix sizes: {matrix_sizes}")
    print(f"Densities: {densities}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Benchmark runs: {benchmark_runs}")
    print()

    # Create results directory
    results_dir = os.path.join(cpu_dir, "benchmark_results")
    os.makedirs(results_dir, exist_ok=True)

    # Results storage
    all_results = []

    # Test each implementation
    for impl in IMPLEMENTATIONS.keys():
        print(f"\n{'='*60}")
        print(f"Testing {IMPLEMENTATIONS[impl]}")
        print(f"{'='*60}")

        # Compile implementation
        print(f"Compiling {impl}...")
        exec_path = compile_impl(impl)
        if exec_path is None:
            print(f"Failed to compile {impl}, skipping...")
            continue

        # Test each matrix size and density
        for size in matrix_sizes:
            for density in densities:
                print(f"\n  Testing {size}x{size} matrix with density {density}...")

                # Generate test matrices
                try:
                    generate_matrices(size, density)
                except subprocess.CalledProcessError as e:
                    print(f"    Failed to generate matrices: {e}")
                    continue

                # Run benchmark
                print(f"    Running benchmark...")
                use_mkl = "human_baseline" in impl
                result = run_spgemm_benchmark(exec_path, warmup_runs, benchmark_runs, use_mkl)

                if result is None:
                    print(f"    Benchmark failed for {size}x{size}, density {density}")
                    continue

                # Load matrix statistics for performance calculation
                try:
                    A_nnz, A_rows, A_cols = load_matrix_stats("A")
                    B_nnz, B_rows, B_cols = load_matrix_stats("B")

                    # Count operations
                    A_indptr, A_indices = load_indptr_indices("A")
                    B_indptr, B_indices = load_indptr_indices("B")
                    total_ops = count_operations(A_indptr, A_indices, B_indptr)
                    total_flops = 2 * total_ops

                    # Calculate GFLOPS
                    gflops = total_flops / (result['mean'] * 1e9)

                    # Store results
                    all_results.append({
                        'implementation': impl,
                        'size': size,
                        'density': density,
                        'time_mean': result['mean'],
                        'time_std': result['std'],
                        'time_min': result['min'],
                        'time_max': result['max'],
                        'gflops': gflops,
                        'operations': total_ops,
                        'total_flops': total_flops
                    })

                    print(f"    Success: {result['mean']:.6f}s ± {result['std']:.6f}s, {gflops:.2f} GFLOPS")

                except Exception as e:
                    print(f"    Failed to calculate performance: {e}")
                    continue

    # Save results to CSV
    if all_results:
        results_df = pd.DataFrame(all_results)
        csv_path = os.path.join(cpu_dir, "benchmark_results.csv")
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

        # Create plots
        plot_path = os.path.join(cpu_dir, "benchmark_results.png")
        plot_results(results_df, plot_path)

        # Print summary
        print(f"\n{'='*60}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*60}")

        for impl in results_df['implementation'].unique():
            impl_data = results_df[results_df['implementation'] == impl]
            print(f"\n{IMPLEMENTATIONS[impl]}:")
            for size in matrix_sizes:
                size_data = impl_data[impl_data['size'] == size]
                if not size_data.empty:
                    print(f"  {size}x{size}: ", end="")
                    for _, row in size_data.iterrows():
                        print(f"{row['density']}: {row['gflops']:.2f} GFLOPS, ", end="")
                    print()
    else:
        print("\nNo successful benchmark results to report.")

if __name__ == "__main__":
    main()