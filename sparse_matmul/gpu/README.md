# Sparse Matrix Multiplication on a GPU

1. In the utils directory, run `python csr_generator.py`
2. In the gpu directory, run `nvcc -std=c++17 main.cpp csr_io.cpp "file-name".cpp -o spgemm_exec -lcusparse` to compile the respective file (replace file-name.cpp with the actual file) and run `./spgemm_exec` to execute it
3. In the utils directory, run `python compare.py` to check correctness
