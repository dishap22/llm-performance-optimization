# Sparse Matrix Multiplication on a CPU

Sample commands run the human benchmark file on pop-os (after installing intel mkl, from the cpu directory)
`source /opt/intel/oneapi/setvars.sh`

`g++ -std=c++17 -I/opt/intel/oneapi/mkl/latest/include main.cpp human_baseline.cpp ../utils/csr_io.cpp -L/opt/intel/oneapi/mkl/latest/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -o spgemm_cpu_exec`

To check correctness (from utils directory):
`python compare.py ../data/py_C_data.txt ../data/py_C_indices.txt ../data/py_C_indptr.txt ../cpu/C_data.txt ../cpu/C_indices.txt ../cpu/C_indptr.txt`
