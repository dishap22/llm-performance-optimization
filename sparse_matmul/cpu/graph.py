import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

IMPLEMENTATIONS = {
    "human_baseline.cpp": "Intel MKL Baseline",
    "chatgpt_2.cpp": "ChatGPT 1st Implementation",
    "gemini_1.cpp": "Gemini 1st Implementation",
    "chatgpt_3.cpp": "ChatGPT 2nd Implementation",
    "gemini_2.cpp": "Gemini 2nd Implementation",
    "chatgpt_4.cpp": "ChatGPT 3rd Implementation",
    "gemini_3.cpp": "Gemini 3rd Implementation"
}

def plot_linear_ticks(results_df, save_path="benchmark_results_linear_ticks.png"):
    plt.style.use('default')
    plt.rcParams['figure.dpi'] = 200
    plt.rcParams['font.family'] = 'monospace'

    if results_df.empty:
        print("No benchmark results available for plotting")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Performance vs Matrix Size (fixed density, linear x, all sizes labeled)
    density = 0.1
    df_size = results_df[results_df['density'] == density]
    sizes = sorted(df_size['size'].unique())
    for impl in df_size['implementation'].unique():
        impl_data = df_size[df_size['implementation'] == impl]
        ax1.plot(impl_data['size'], impl_data['gflops'], marker='o',
                 label=IMPLEMENTATIONS.get(impl, impl), linewidth=2, markersize=6)
    ax1.set_xlabel('Matrix Size (N x N)')
    ax1.set_ylabel('GFLOPS')
    ax1.set_title(f'CPU SpGEMM Performance vs Matrix Size (Density: {density})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('linear')
    ax1.set_yscale('linear')
    ax1.set_xticks(sizes)
    ax1.set_xticklabels([str(s) for s in sizes])

    # Plot 2: Performance vs Density (fixed size, linear x, all densities labeled)
    size = 1024
    df_density = results_df[results_df['size'] == size]
    densities = sorted(df_density['density'].unique())
    for impl in df_density['implementation'].unique():
        impl_data = df_density[df_density['implementation'] == impl]
        ax2.plot(impl_data['density'], impl_data['gflops'], marker='s',
                 label=IMPLEMENTATIONS.get(impl, impl), linewidth=2, markersize=6)
    ax2.set_xlabel('Matrix Density')
    ax2.set_ylabel('GFLOPS')
    ax2.set_title(f'CPU SpGEMM Performance vs Density (Size: {size}x{size})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('linear')
    ax2.set_yscale('linear')
    ax2.set_xticks(densities)
    ax2.set_xticklabels([str(d) for d in densities])

    # Plot 3: Relative Performance (normalized to MKL, fixed size/density)
    baseline = "human_baseline.cpp"
    df_rel = results_df.copy()
    for (size, density), group in df_rel.groupby(['size', 'density']):
        baseline_group = group[group['implementation'] == baseline]
        if not baseline_group.empty:
            baseline_perf = baseline_group['gflops'].iloc[0]
            if baseline_perf > 0:
                df_rel.loc[group.index, 'relative_perf'] = group['gflops'] / baseline_perf
    df_rel_plot = df_rel[(df_rel['size'] == 1024) & (df_rel['density'] == 0.1)]
    if not df_rel_plot.empty:
        implementations = [impl for impl in df_rel_plot['implementation'].unique() if impl != baseline]
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
            bars = ax3.bar(impl_names, rel_perfs, color=['#ff7f0e', '#2ca02c', '#1f77b4', '#d62728'][:len(rel_perfs)])
            ax3.set_ylabel('Relative Performance (MKL = 1.0)')
            ax3.set_title('Relative Performance Comparison')
            ax3.grid(True, alpha=0.3, axis='y')
            for bar, perf in zip(bars, rel_perfs):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                         f'{perf:.2f}x', ha='center', va='bottom')
        else:
            ax3.text(0.5, 0.5, 'No relative performance data available',
                     ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Relative Performance Comparison')
    else:
        ax3.text(0.5, 0.5, 'No data for size=1024, density=0.1',
                 ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Relative Performance Comparison')

    # Plot 4: Memory Efficiency (GFLOPS/GB, fixed size, linear x, all densities labeled)
    df_mem = results_df[results_df['size'] == 1024].copy()
    df_mem['memory_gb'] = df_mem['size'] * df_mem['size'] * 4 * df_mem['density'] * 2 / 1e9  # Rough estimate
    df_mem['gflops_per_gb'] = df_mem['gflops'] / df_mem['memory_gb']
    for impl in df_mem['implementation'].unique():
        impl_data = df_mem[df_mem['implementation'] == impl]
        ax4.plot(impl_data['density'], impl_data['gflops_per_gb'], marker='^',
                 label=IMPLEMENTATIONS.get(impl, impl), linewidth=2, markersize=6)
    ax4.set_xlabel('Matrix Density')
    ax4.set_ylabel('GFLOPS/GB')
    ax4.set_title('Memory Efficiency (Size: 1024x1024)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('linear')
    ax4.set_yscale('linear')
    ax4.set_xticks(densities)
    ax4.set_xticklabels([str(d) for d in densities])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Results plotted and saved to {save_path}")

if __name__ == "__main__":
    csv_path = "benchmark_results.csv"
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found.")
    else:
        df = pd.read_csv(csv_path)
        plot_linear_ticks(df)