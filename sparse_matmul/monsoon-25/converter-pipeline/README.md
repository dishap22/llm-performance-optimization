gcc -c mmio.c -o mmio.o
g++ -O3 -std=c++17 mtx_to_csr.cpp mmio.o -o mtx_to_csr
./mtx_to_csr ../data/atmosmodd/atmosmodd.mtx ../data/atmosmodd/atmosmodd
./mtx_to_csr ../data/G3_circuit/G3_circuit.mtx ../data/G3_circuit/G3_circuit
./mtx_to_csr ../data/stokes/stokes.mtx ../data/stokes/stokes
./mtx_to_csr ../data/webbase-1M/webbase-1M.mtx ../data/webbase-1M/webbase-1M
./mtx_to_csr ../data/bcsstk36/bcsstk36.mtx ../data/bcsstk36/bcsstk36
./mtx_to_csr ../data/soc-Pokec/soc-Pokec.mtx ../data/soc-Pokec/soc-Pokec
./mtx_to_csr ../data/wiki-Vote/wiki-Vote.mtx ../data/wiki-Vote/wiki-Vote
./mtx_to_csr ../data/com-Amazon/com-Amazon.mtx ../data/com-Amazon/com-Amazon
./mtx_to_csr ../data/email-Enron/email-Enron.mtx ../data/email-Enron/email-Enron
./mtx_to_csr ../data/roadNet-CA/roadNet-CA.mtx ../data/roadNet-CA/roadNet-CA
