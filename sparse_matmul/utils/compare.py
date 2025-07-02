import numpy as np
from scipy.sparse import csr_matrix
import sys
import os

def load_csr_matrix(data_file, indices_file, indptr_file):
    try:
        data = np.loadtxt(data_file, dtype=np.float64)
        indices = np.loadtxt(indices_file, dtype=int)
        indptr = np.loadtxt(indptr_file, dtype=int)

        # Prevent single value from being treated as a scalar
        if data.ndim == 0:
            data = np.array([data])
        if indices.ndim == 0:
            indices = np.array([indices])
        if indptr.ndim == 0:
            indptr = np.array([indptr])

        return data, indices, indptr
    except Exception as e:
        print(f"Error loading matrix from {data_file}, {indices_file}, {indptr_file}: {e}")
        return None, None, None

def matrices_equal(data1, indices1, indptr1, data2, indices2, indptr2, tolerance=1e-4):
    if len(indptr1) != len(indptr2):
        return False, f"Different number of rows: {len(indptr1)-1} vs {len(indptr2)-1}", None

    n_rows = len(indptr1) - 1

    try:
        max_col1 = max(indices1) if len(indices1) > 0 else 0
        max_col2 = max(indices2) if len(indices2) > 0 else 0
        n_cols = max(max_col1, max_col2) + 1

        shape = (n_rows, n_cols)

        matrix1 = csr_matrix((data1, indices1, indptr1), shape=shape)
        matrix2 = csr_matrix((data2, indices2, indptr2), shape=shape)
        dense1 = matrix1.toarray()
        dense2 = matrix2.toarray()

        diff_matrix = np.abs(dense1 - dense2)
        max_diff = np.max(diff_matrix)
        significant_diffs = np.sum(diff_matrix > tolerance)
        total_elements = dense1.size

        if max_diff <= tolerance:
            return True, f"Matrices are equal (max difference: {max_diff:.2e})", max_diff
        else:
            return False, f"Matrices differ: {significant_diffs}/{total_elements} elements exceed tolerance (max difference: {max_diff:.2e})", max_diff

    except Exception as e:
        return False, f"Error during comparison: {e}", None

def print_matrix_info(data, indices, indptr, name):
    n_rows = len(indptr) - 1
    n_cols = max(indices) + 1 if len(indices) > 0 else 1
    nnz = len(data)

    print(f"{name}:")
    print(f"  Shape: {n_rows} x {n_cols}")
    print(f"  Non-zeros: {nnz}")
    if nnz > 0:
        print(f"  Data range: [{np.min(data):.6f}, {np.max(data):.6f}]")
    print()

def main():
    if len(sys.argv) != 7:
        print("Usage: python compare.py <data1> <indices1> <indptr1> <data2> <indices2> <indptr2>\n")
        print("Example:\n  python compare.py ../data/py_C_data.txt ../data/py_C_indices.txt ../data/py_C_indptr.txt ../gpu/C_data.txt ../gpu/C_indices.txt ../gpu/C_indptr.txt")
        sys.exit(1)

    data1_file, indices1_file, indptr1_file = sys.argv[1:4]
    data2_file, indices2_file, indptr2_file = sys.argv[4:7]

    files_to_check = [data1_file, indices1_file, indptr1_file, data2_file, indices2_file, indptr2_file]
    for file in files_to_check:
        if not os.path.exists(file):
            print(f"Error: File not found: {file}")
            sys.exit(1)

    data1, indices1, indptr1 = load_csr_matrix(data1_file, indices1_file, indptr1_file)
    if data1 is None:
        sys.exit(1)

    data2, indices2, indptr2 = load_csr_matrix(data2_file, indices2_file, indptr2_file)
    if data2 is None:
        sys.exit(1)

    print()
    print_matrix_info(data1, indices1, indptr1, "Matrix 1")
    print_matrix_info(data2, indices2, indptr2, "Matrix 2")

    is_equal, message, max_diff = matrices_equal(data1, indices1, indptr1, data2, indices2, indptr2)

    print(f"Result: {message}")

    if is_equal:
        print("SUCCESS: The matrices are equivalent")
        if max_diff is not None and max_diff > 0:
            print(f"   (Small numerical differences within tolerance: {max_diff:.2e})")
    else:
        print("FAILURE: The matrices are different")
        if len(data1) == len(data2):
            print(f"\nFirst 5 data values:")
            print(f"Matrix 1: {data1[:5]}")
            print(f"Matrix 2: {data2[:5]}")

            print(f"\nFirst 5 indices:")
            print(f"Matrix 1: {indices1[:5]}")
            print(f"Matrix 2: {indices2[:5]}")

            if len(data1) >= 5 and len(data2) >= 5:
                data_diff = data1[:5] - data2[:5]
                print(f"\nFirst 5 data differences:")
                print(f"Diff:     {data_diff}")

            print(f"\nMatrix statistics:")
            print(f"Matrix 1 - Mean: {np.mean(data1):.6f}, Std: {np.std(data1):.6f}")
            print(f"Matrix 2 - Mean: {np.mean(data2):.6f}, Std: {np.std(data2):.6f}")

if __name__ == "__main__":
    main()