import pandas as pd
import io
import matplotlib.pyplot as plt

try:
    df = pd.read_csv('cores.csv')
    # The CSV might have leading spaces in column names, so we strip them.
    df.columns = df.columns.str.strip()
except FileNotFoundError:
    print("Error: 'cores.csv' not found. Please make sure the file is in the same directory.")
    exit()

# Plot Execution Time vs. Cores
plt.figure(figsize=(10, 6))
plt.plot(df['Cores'], df['ExecutionTime_s'], marker='o')
plt.title('Execution Time vs. Cores')
plt.xlabel('Cores')
plt.ylabel('Execution Time (s)')
plt.grid(True)
plt.savefig('execution_time_vs_cores.png')
plt.close()

# Plot GFLOPS vs. Cores
plt.figure(figsize=(10, 6))
plt.plot(df['Cores'], df['GFLOPS'], marker='o', color='green')
plt.title('GFLOPS vs. Cores')
plt.xlabel('Cores')
plt.ylabel('GFLOPS')
plt.grid(True)
plt.savefig('gflops_vs_cores.png')
plt.close()

print("Plots saved as execution_time_vs_cores.png and gflops_vs_cores.png")