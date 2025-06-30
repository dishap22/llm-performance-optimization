import numpy as np
import argparse
import os
from scipy.sparse import random

def generate_and_save_csr(matrix_name, matrix, output_dir='.'):
    os.makedirs(output_dir, exist_ok=True)

    matrix = matrix.tocsr()
    np.savetxt(f'{output_dir}/py_{matrix_name}_indptr.txt', matrix.indptr, fmt='%d')
    np.savetxt(f'{output_dir}/py_{matrix_name}_indices.txt', matrix.indices, fmt='%d')
    np.savetxt(f'{output_dir}/py_{matrix_name}_data.txt', matrix.data, fmt='%.6f')
    print(f"Saved CSR components of {matrix_name} in {output_dir}/")

def main():
    parser = argparse.ArgumentParser(description='Generate sparse CSR matrices')
    parser.add_argument('--rows', type=int, default=1000, help='Number of rows in the matrix (default: 1000)')
    parser.add_argument('--cols', type=int, default=1000, help='Number of columns in the matrix (default: 1000)')
    parser.add_argument('--density', type=float, default=0.2, help='Density of the sparse matrix (default: 0.2)')
    parser.add_argument('--output-dir', type=str, default='../data', help='Output directory for matrix files (default: ../data)')

    args = parser.parse_args()

    shape = (args.rows, args.cols)
    density = args.density

    print(f"Generating sparse matrices with shape {shape} and density {density}")

    A = random(*shape, density=density, format='csr', dtype=np.float32)
    B = random(*shape, density=density, format='csr', dtype=np.float32)
    C = A @ B

    output_dir = args.output_dir
    generate_and_save_csr('A', A, output_dir)
    generate_and_save_csr('B', B, output_dir)
    generate_and_save_csr('C', C, output_dir)

if __name__ == "__main__":
    main()
