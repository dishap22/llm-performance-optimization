# Sparse Matrix Multiplication on a GPU

1. In the utils directory, run `python csr_generator.py`
2. In the gpu directory, run `nvcc -std=c++17 main.cpp ../utils/csr_io.cpp "spgemm-file" -o spgemm_exec -lcusparse` to compile the respective file (replace "spgemm-file" with the actual file) and run `./spgemm_exec` to execute it
3. In the utils directory, run `python compare.py ../data/py_C_data.txt ../data/py_C_indices.txt ../data/py_C_indptr.txt ../gpu/C_data.txt ../gpu/C_indices.txt ../gpu/C_indptr.txt` to check correctness

Sample Commands:

`nvcc -std=c++17 main.cpp ../utils/csr_io.cpp human_baseline.cpp -o spgemm_exec -lcusparse`

`nvcc -std=c++17 main.cpp ../utils/csr_io.cpp gemini_1.cu -o spgemm_exec`

`nvcc -std=c++17 main.cpp ../utils/csr_io.cpp chatgpt_1.cu -o spgemm_exec`

Profiling done using Nsight Compute, for e.g. `sudo ~/tools/nsight-compute/ncu ./spgemm_exec`
