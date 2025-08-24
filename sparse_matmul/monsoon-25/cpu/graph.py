import pandas as pd
import matplotlib.pyplot as plt

# --- Read the data from the CSV file ---
try:
    df = pd.read_csv('results_averaged.csv')
    # The CSV might have leading spaces in column names, so we strip them.
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    print("Error: 'results_averaged.csv' not found. Please make sure the file is in the same directory.")
    exit()

# ===================================================================
# Plot 1: Execution Time vs. NNZ (Log Scale)
# ===================================================================
df_nnz_sorted = df.sort_values('NNZ_A')

plt.figure(figsize=(12, 8))
plt.scatter(df_nnz_sorted['NNZ_A'], df_nnz_sorted['ExecutionTime_s'], marker='o')

# Annotate each point with the matrix name
for i, row in df_nnz_sorted.iterrows():
    plt.annotate(row['MatrixName'],
                 (row['NNZ_A'], row['ExecutionTime_s']),
                 textcoords="offset points",
                 xytext=(5,5), # Offset the text by 5 points in x and y
                 ha='left',    # Horizontal alignment
                 fontsize=9)

plt.xscale('log')
plt.yscale('log')
plt.title('Execution Time vs. Number of Non-Zeros (NNZ) - Log Scale', fontsize=16)
plt.xlabel('Number of Non-Zeros (NNZ) - Log Scale', fontsize=12)
plt.ylabel('Execution Time (seconds) - Log Scale', fontsize=12)
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('time_vs_nnz_log_annotated.png')

# ===================================================================
# Plot 2: Execution Time vs. Rows (Log Scale)
# ===================================================================
df_rows_sorted = df.sort_values('Rows_A')

plt.figure(figsize=(12, 8))
plt.scatter(df_rows_sorted['Rows_A'], df_rows_sorted['ExecutionTime_s'], marker='s', color='green')

# Annotate each point with the matrix name
for i, row in df_rows_sorted.iterrows():
    plt.annotate(row['MatrixName'],
                 (row['Rows_A'], row['ExecutionTime_s']),
                 textcoords="offset points",
                 xytext=(5,5), # Offset the text by 5 points in x and y
                 ha='left',    # Horizontal alignment
                 fontsize=9)

plt.xscale('log')
plt.yscale('log')
plt.title('Execution Time vs. Number of Rows (Vertices) - Log Scale', fontsize=16)
plt.xlabel('Number of Rows - Log Scale', fontsize=12)
plt.ylabel('Execution Time (seconds) - Log Scale', fontsize=12)
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('time_vs_rows_log_annotated.png')

# ===================================================================
# Plot 3: GFLOPS vs. NNZ (Log Scale for X-axis)
# ===================================================================
plt.figure(figsize=(12, 8))
plt.scatter(df_nnz_sorted['NNZ_A'], df_nnz_sorted['GFLOPS'], marker='o', color='red')

# Annotate each point with the matrix name
for i, row in df_nnz_sorted.iterrows():
    plt.annotate(row['MatrixName'],
                 (row['NNZ_A'], row['GFLOPS']),
                 textcoords="offset points",
                 xytext=(5,5),
                 ha='left',
                 fontsize=9)

plt.xscale('log')
plt.title('Performance (GFLOPS) vs. Number of Non-Zeros (NNZ)', fontsize=16)
plt.xlabel('Number of Non-Zeros (NNZ) - Log Scale', fontsize=12)
plt.ylabel('Performance Rate (GFLOPS) - Linear Scale', fontsize=12)
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('gflops_vs_nnz_log.png')

# ===================================================================
# Plot 4: GFLOPS vs. Rows (Log Scale for X-axis)
# ===================================================================
plt.figure(figsize=(12, 8))
plt.scatter(df_rows_sorted['Rows_A'], df_rows_sorted['GFLOPS'], marker='s', color='purple')

# Annotate each point with the matrix name
for i, row in df_rows_sorted.iterrows():
    plt.annotate(row['MatrixName'],
                 (row['Rows_A'], row['GFLOPS']),
                 textcoords="offset points",
                 xytext=(5,5),
                 ha='left',
                 fontsize=9)

plt.xscale('log')
plt.title('Performance (GFLOPS) vs. Number of Rows (Vertices)', fontsize=16)
plt.xlabel('Number of Rows - Log Scale', fontsize=12)
plt.ylabel('Performance Rate (GFLOPS) - Linear Scale', fontsize=12)
plt.grid(True, which="both", ls="--")
plt.tight_layout()
plt.savefig('gflops_vs_rows_log.png')


print("Generated four graph images.")