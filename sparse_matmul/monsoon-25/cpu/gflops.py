import sys


nnz_c = float(sys.argv[1])
execution_time = float(sys.argv[2])

print(f"nnz_c: {nnz_c}")
print(f"execution_time: {execution_time}")
print(f"gflops: {2 * nnz_c / execution_time / 1e9}")