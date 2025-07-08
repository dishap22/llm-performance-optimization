# Sparse Matrix Multiplication on a CPU

Sample commands to compile the human benchmark file on pop-os (after installing intel mkl, from the cpu directory)
`source /opt/intel/oneapi/setvars.sh`

`export MKL_NUM_THREADS=$(nproc) export OMP_NUM_THREADS=$(nproc)`

`g++ -std=c++17 -I/opt/intel/oneapi/mkl/latest/include main.cpp human_baseline.cpp ../utils/csr_io.cpp -L/opt/intel/oneapi/mkl/latest/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl -o spgemm_cpu_exec`

To check correctness (from utils directory):
`python compare.py ../data/py_C_data.txt ../data/py_C_indices.txt ../data/py_C_indptr.txt ../cpu/C_data.txt ../cpu/C_indices.txt ../cpu/C_indptr.txt`

Sample commands to compile the LLM written code:
`g++ -std=c++17 -fopenmp -I../utils main.cpp gemini_1.cpp ../utils/csr_io.cpp  -o spgemm_cpu_exec`

`g++ -std=c++17 -fopenmp -I../utils main.cpp chatgpt_2.cpp ../utils/csr_io.cpp -o spgemm_cpu_exec`

`g++ -std=c++17 -fopenmp -I../utils main.cpp gemini_2.cpp ../utils/csr_io.cpp  -o spgemm_cpu_exec`

`g++ -std=c++17 -fopenmp -I../utils main.cpp chatgpt_3.cpp ../utils/csr_io.cpp -o spgemm_cpu_exec`


Perf:

```
perf record ./spgemm_cpu_exec
perf report
perf annotate
perf stat -e cycles,instructions,cache-references,cache-misses,branches,branch-misses ./spgemm_cpu_exec
```

Valgrind/Callgrind:

```
valgrind --tool=callgrind ./spgemm_cpu_exec
callgrind_annotate callgrind.out.74326 | less > chatgpt_callgrind1.txt
valgrind --leak-check=full --show-leak-kinds=all --track-origins=yes ./spgemm_cpu_exec 2> chatgpt_valgrind1.txt
```
