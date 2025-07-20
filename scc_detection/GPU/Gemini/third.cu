#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <algorithm>
#include <chrono>
#include <numeric>

// CUDA runtime API
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__,     \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Define sparse/dense threshold (adjust based on profiling)
// A common heuristic is around E/20 or a fixed value like 2048 or 4096
#define SPARSE_THRESHOLD 2048

// --- Device Utilities for Bitsets ---

// Get the number of unsigned long longs needed for a bitset of 'size' bits
__host__ __device__ int get_ull_size(int num_bits) {
    if (num_bits <= 0) return 0; // Handle empty bitset
    return (num_bits + 63) / 64;
}

// Check if a bit is set in a bitset (DEVICE function)
__device__ bool is_bit_set(const unsigned long long* bitset, int bit_idx) {
    // Add defensive check for bitset pointer
    if (!bitset || bit_idx < 0) return false;
    return (bitset[bit_idx / 64] & (1ULL << (bit_idx % 64)));
}

// Atomically set a bit in a bitset (DEVICE function)
__device__ bool atomic_set_bit(unsigned long long* bitset, int bit_idx) {
    // Add defensive check for bitset pointer
    if (!bitset || bit_idx < 0) return false;
    unsigned long long old_val = atomicOr(&bitset[bit_idx / 64], (1ULL << (bit_idx % 64)));
    return ! (old_val & (1ULL << (bit_idx % 64))); // True if bit was not set before
}

// Host-side function to check if a bit is set in a vector<unsigned long long> representing a bitset
// Used when copying bitset to host to check individual bits.
bool host_is_bit_set(const std::vector<unsigned long long>& bitset_vec, int bit_idx) {
    if (bit_idx < 0 || bit_idx >= (int)bitset_vec.size() * 64) return false; // Bounds check
    return (bitset_vec[bit_idx / 64] & (1ULL << (bit_idx % 64)));
}

// --- Small Helper Kernels for Host-to-Device Bit/Value Setting ---
// These are needed because __device__ functions cannot be called from __host__
__global__ void setBitKernel(unsigned long long* d_bitset, int bit_idx) {
    if (d_bitset && bit_idx >= 0) { // Defensive check
        atomic_set_bit(d_bitset, bit_idx);
    }
}

__global__ void assignSccIdKernel(int* d_scc_id_array, int node_idx, int scc_id) {
    // This kernel should typically be launched with 1 block, 1 thread
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        if (d_scc_id_array && node_idx >= 0) { // Defensive check
            atomicExch(&d_scc_id_array[node_idx], scc_id);
        }
    }
}


// Kernel to initialize an integer array on the device
__global__ void initializeArrayKernel(int* d_array, int value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = value;
    }
}

// Kernel to initialize a bitset (unsigned long long array) to all zeros
__global__ void initializeBitsetKernel(unsigned long long* d_bitset, int ull_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ull_size) {
        d_bitset[idx] = 0ULL;
    }
}

// Kernel to compute initial in-degrees and out-degrees
__global__ void computeDegreesKernel(int num_nodes, const int* d_adj_list_starts,
                                    const int* d_transposed_adj_list_starts,
                                    int* d_out_degree, int* d_in_degree) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (node_idx < num_nodes) {
        d_out_degree[node_idx] = d_adj_list_starts[node_idx + 1] - d_adj_list_starts[node_idx];
        d_in_degree[node_idx] = d_transposed_adj_list_starts[node_idx + 1] - d_transposed_adj_list_starts[node_idx];
    }
}

// Kernel for the trimming step
__global__ void trimGraphKernel(int num_nodes,
                                const int* d_adj_list_starts, const int* d_adj_list_edges,
                                const int* d_transposed_adj_list_starts, const int* d_transposed_adj_list_edges,
                                int* d_out_degree, int* d_in_degree,
                                int* d_node_status, // 0: active, 1: trimmed
                                int* d_scc_id,
                                int current_scc_id_base, // Base for new SCC IDs
                                int* d_trimmed_count // Atomic counter for newly trimmed nodes in this iter
                                ) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_idx < num_nodes) {
        // Only process active nodes (status 0)
        if (d_node_status[node_idx] == 0) {
            if (d_out_degree[node_idx] == 0 || d_in_degree[node_idx] == 0) {
                // Atomically assign SCC ID and mark as trimmed
                int old_status = atomicCAS(&d_node_status[node_idx], 0, 1);
                if (old_status == 0) { // If successfully marked as trimmed
                    atomicExch(&d_scc_id[node_idx], current_scc_id_base + atomicAdd(d_trimmed_count, 1));

                    // Decrement degrees of neighbors
                    if (d_out_degree[node_idx] == 0) { // It's a source node
                        int start_edge = d_transposed_adj_list_starts[node_idx];
                        int end_edge = d_transposed_adj_list_starts[node_idx + 1];
                        for (int i = start_edge; i < end_edge; ++i) {
                            int neighbor = d_transposed_adj_list_edges[i];
                            if (d_node_status[neighbor] == 0) { // Only affect active neighbors
                                atomicSub(&d_out_degree[neighbor], 1);
                            }
                        }
                    }

                    if (d_in_degree[node_idx] == 0) { // It's a sink node
                        int start_edge = d_adj_list_starts[node_idx];
                        int end_edge = d_adj_list_starts[node_idx + 1];
                        for (int i = start_edge; i < end_edge; ++i) {
                            int neighbor = d_adj_list_edges[i];
                            if (d_node_status[neighbor] == 0) { // Only affect active neighbors
                                atomicSub(&d_in_degree[neighbor], 1);
                            }
                        }
                    }
                }
            }
        }
    }
}

// Kernel to build the compacted graph (active nodes only)
__global__ void compactGraphKernel(int num_nodes_original, // Number of nodes in the original graph
                                   const int* d_old_adj_list_starts, const int* d_old_adj_list_edges,
                                   const int* d_node_status, // 0: active, 1: trimmed
                                   const int* d_active_node_map, // old_node_id -> new_compacted_id
                                   int* d_new_adj_list_starts, // CSR starts for compacted graph
                                   int* d_new_adj_list_edges, // CSR edges for compacted graph
                                   int* d_new_total_edges_counter // Atomic counter for total edges in compacted graph
                                   ) {
    int old_node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (old_node_idx < num_nodes_original) {
        if (d_node_status[old_node_idx] == 0) { // If this node is active
            int new_node_idx = d_active_node_map[old_node_idx];

            int start_edge = d_old_adj_list_starts[old_node_idx];
            int end_edge = d_old_adj_list_starts[old_node_idx + 1];

            // Count edges for this specific node in the new compacted graph
            int current_new_node_edge_count = 0;
            for (int i = start_edge; i < end_edge; ++i) {
                int old_neighbor = d_old_adj_list_edges[i];
                if (d_node_status[old_neighbor] == 0) { // Only include edges to active neighbors
                    current_new_node_edge_count++;
                }
            }

            // Get the starting position for this new_node_idx's edges atomically
            int edge_write_start_pos = atomicAdd(d_new_total_edges_counter, current_new_node_edge_count);
            d_new_adj_list_starts[new_node_idx] = edge_write_start_pos;

            // Fill edges.
            int current_write_idx = edge_write_start_pos;
            for (int i = start_edge; i < end_edge; ++i) {
                int old_neighbor = d_old_adj_list_edges[i];
                if (d_node_status[old_neighbor] == 0) {
                    d_new_adj_list_edges[current_write_idx++] = d_active_node_map[old_neighbor];
                }
            }
        }
    }
}


// Optimized BFS Kernel for Kosaraju's First Pass (sparse mode)
// Processes a frontier given as a list of node IDs.
__global__ void bfs_sparse_kernel(int frontier_size, // Number of nodes in d_sparse_frontier
                                  const int* d_adj_list_starts, const int* d_adj_list_edges,
                                  unsigned long long* d_visited_bfs_state, // Visited for this specific BFS
                                  unsigned long long* d_global_visited_first_pass, // Visited for first pass overall
                                  const int* d_sparse_frontier, // Input frontier (list of nodes)
                                  int* d_sparse_next_frontier,  // Output frontier (list of nodes)
                                  int* d_sparse_next_frontier_count // Atomic counter for d_sparse_next_frontier
                                  ) {
    int frontier_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (frontier_idx < frontier_size) {
        int node = d_sparse_frontier[frontier_idx];

        int start_edge = d_adj_list_starts[node];
        int end_edge = d_adj_list_starts[node + 1];

        for (int i = start_edge; i < end_edge; ++i) {
            int neighbor = d_adj_list_edges[i];

            // If neighbor not visited in this BFS and not globally finished
            if (!is_bit_set(d_visited_bfs_state, neighbor) && !is_bit_set(d_global_visited_first_pass, neighbor)) {
                if (atomic_set_bit(d_visited_bfs_state, neighbor)) { // Atomically mark visited in this BFS
                    // Atomically mark globally visited for first pass
                    atomic_set_bit(d_global_visited_first_pass, neighbor);
                    int new_pos = atomicAdd(d_sparse_next_frontier_count, 1);
                    d_sparse_next_frontier[new_pos] = neighbor;
                }
            }
        }
    }
}

// Optimized BFS Kernel for Kosaraju's First Pass (dense mode using bitset)
// Iterates through all nodes and checks if they are in the current frontier bitset.
__global__ void bfs_dense_kernel(int num_active_nodes, // Total active nodes in compacted graph
                                 const int* d_adj_list_starts, const int* d_adj_list_edges,
                                 unsigned long long* d_visited_bfs_state,
                                 unsigned long long* d_global_visited_first_pass,
                                 unsigned long long* d_current_frontier_bitset,
                                 unsigned long long* d_next_frontier_bitset,
                                 int* d_new_nodes_found_count // Total newly found nodes for next frontier
                                 ) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < num_active_nodes) {
        // If this node is in the current frontier
        if (is_bit_set(d_current_frontier_bitset, node)) {
            int start_edge = d_adj_list_starts[node];
            int end_edge = d_adj_list_starts[node + 1];

            for (int i = start_edge; i < end_edge; ++i) {
                int neighbor = d_adj_list_edges[i];

                if (!is_bit_set(d_visited_bfs_state, neighbor) && !is_bit_set(d_global_visited_first_pass, neighbor)) {
                    if (atomic_set_bit(d_visited_bfs_state, neighbor)) {
                        atomic_set_bit(d_global_visited_first_pass, neighbor);
                        if (atomic_set_bit(d_next_frontier_bitset, neighbor)) { // If successfully added to next frontier
                             atomicAdd(d_new_nodes_found_count, 1);
                        }
                    }
                }
            }
        }
    }
}

// Kernel to mark nodes as finished in the finishing order
// This kernel takes the *old* current_frontier_bitset, as these are the nodes whose exploration just completed.
__global__ void markFinishedKernel(int num_active_nodes,
                                   int* d_finishing_order, int* d_current_time,
                                   unsigned long long* d_finished_frontier_bitset // The frontier that just finished being processed
                                   ) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;
    if (node < num_active_nodes) {
        // If this node was part of the frontier that just finished
        if (is_bit_set(d_finished_frontier_bitset, node)) {
            // This node just finished being part of a BFS level, record its finishing order
            int time = atomicAdd(d_current_time, 1);
            d_finishing_order[num_active_nodes - 1 - time] = node;
        }
    }
}


// Optimized BFS Kernel for Kosaraju's Second Pass (sparse mode)
__global__ void scc_sparse_kernel(int frontier_size,
                                  const int* d_transposed_adj_list_starts, const int* d_transposed_adj_list_edges,
                                  unsigned long long* d_visited_bfs_state, // Visited for this specific BFS
                                  unsigned long long* d_global_visited_second_pass, // Visited for second pass overall
                                  int* d_scc_id, int current_scc_id, // For assigning SCC IDs
                                  const int* d_sparse_frontier,
                                  int* d_sparse_next_frontier,
                                  int* d_sparse_next_frontier_count
                                  ) {
    int frontier_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (frontier_idx < frontier_size) {
        int node = d_sparse_frontier[frontier_idx];

        int start_edge = d_transposed_adj_list_starts[node];
        int end_edge = d_transposed_adj_list_starts[node + 1];

        for (int i = start_edge; i < end_edge; ++i) {
            int neighbor = d_transposed_adj_list_edges[i];

            if (!is_bit_set(d_visited_bfs_state, neighbor) && !is_bit_set(d_global_visited_second_pass, neighbor)) {
                if (atomic_set_bit(d_visited_bfs_state, neighbor)) {
                    atomic_set_bit(d_global_visited_second_pass, neighbor);
                    // Pass the compacted SCC ID, it will be mapped back later
                    atomicExch(&d_scc_id[neighbor], current_scc_id); // Assign SCC ID directly to compacted node
                    int new_pos = atomicAdd(d_sparse_next_frontier_count, 1);
                    d_sparse_next_frontier[new_pos] = neighbor;
                }
            }
        }
    }
}

// Optimized BFS Kernel for Kosaraju's Second Pass (dense mode using bitset)
__global__ void scc_dense_kernel(int num_active_nodes,
                                 const int* d_transposed_adj_list_starts, const int* d_transposed_adj_list_edges,
                                 unsigned long long* d_visited_bfs_state,
                                 unsigned long long* d_global_visited_second_pass,
                                 int* d_scc_id, int current_scc_id,
                                 unsigned long long* d_current_frontier_bitset,
                                 unsigned long long* d_next_frontier_bitset,
                                 int* d_new_nodes_found_count
                                 ) {
    int node = blockIdx.x * blockDim.x + threadIdx.x;

    if (node < num_active_nodes) {
        if (is_bit_set(d_current_frontier_bitset, node)) {
            int start_edge = d_transposed_adj_list_starts[node];
            int end_edge = d_transposed_adj_list_starts[node + 1];

            for (int i = start_edge; i < end_edge; ++i) {
                int neighbor = d_transposed_adj_list_edges[i];

                if (!is_bit_set(d_visited_bfs_state, neighbor) && !is_bit_set(d_global_visited_second_pass, neighbor)) {
                    if (atomic_set_bit(d_visited_bfs_state, neighbor)) {
                        atomic_set_bit(d_global_visited_second_pass, neighbor);
                        atomicExch(&d_scc_id[neighbor], current_scc_id); // Assign SCC ID directly to compacted node
                        if (atomic_set_bit(d_next_frontier_bitset, neighbor)) {
                            atomicAdd(d_new_nodes_found_count, 1);
                        }
                    }
                }
            }
        }
    }
}

// Kernel to map SCC IDs from compacted graph back to original graph nodes
__global__ void mapSccIdsBackKernel(int num_active_nodes, const int* d_compacted_scc_ids,
                                    const int* d_reverse_node_map, int* d_original_scc_ids) {
    int compact_node_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (compact_node_idx < num_active_nodes) {
        int original_node_idx = d_reverse_node_map[compact_node_idx];
        d_original_scc_ids[original_node_idx] = d_compacted_scc_ids[compact_node_idx];
    }
}


// --- Host Code ---

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file.txt>" << std::endl;
        return 1;
    }

    std::ifstream inputFile(argv[1]);
    if (!inputFile.is_open()) {
        std::cerr << "Error opening the file!" << std::endl;
        return 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    int V_original, E_original;
    inputFile >> V_original >> E_original;

    std::vector<std::vector<int>> h_adj(V_original);
    std::vector<std::vector<int>> h_adj_transposed(V_original);

    int u, v_node;
    for (int i = 0; i < E_original; ++i) {
        inputFile >> u >> v_node;
        h_adj[u].push_back(v_node);
        h_adj_transposed[v_node].push_back(u);
    }
    inputFile.close();

    // Convert to CSR for initial graph
    std::vector<int> h_adj_list_starts(V_original + 1, 0);
    std::vector<int> h_adj_list_edges(E_original);
    int current_edge_idx = 0;
    for (int i = 0; i < V_original; ++i) {
        h_adj_list_starts[i] = current_edge_idx;
        for (int neighbor : h_adj[i]) {
            h_adj_list_edges[current_edge_idx++] = neighbor;
        }
    }
    h_adj_list_starts[V_original] = current_edge_idx;

    std::vector<int> h_transposed_adj_list_starts(V_original + 1, 0);
    std::vector<int> h_transposed_adj_list_edges(E_original);
    current_edge_idx = 0;
    for (int i = 0; i < V_original; ++i) {
        h_transposed_adj_list_starts[i] = current_edge_idx;
        for (int neighbor : h_adj_transposed[i]) {
            h_transposed_adj_list_edges[current_edge_idx++] = neighbor;
        }
    }
    h_transposed_adj_list_starts[V_original] = current_edge_idx;

    // --- Device Memory Allocation (for original graph & trimming) ---
    int* d_adj_list_starts_orig;
    int* d_adj_list_edges_orig;
    int* d_transposed_adj_list_starts_orig;
    int* d_transposed_adj_list_edges_orig;
    int* d_out_degree;
    int* d_in_degree;
    int* d_node_status; // 0: active, 1: trimmed
    int* d_scc_id_original; // Final SCC IDs for all original nodes
    int* d_trimmed_count_gpu; // For atomic counting in trimming

    CUDA_CHECK(cudaMalloc(&d_adj_list_starts_orig, (V_original + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_adj_list_edges_orig, E_original * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_transposed_adj_list_starts_orig, (V_original + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_transposed_adj_list_edges_orig, E_original * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_degree, V_original * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_in_degree, V_original * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_status, V_original * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scc_id_original, V_original * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_trimmed_count_gpu, sizeof(int)));

    // Data Transfer (Host to Device)
    CUDA_CHECK(cudaMemcpy(d_adj_list_starts_orig, h_adj_list_starts.data(), (V_original + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adj_list_edges_orig, h_adj_list_edges.data(), E_original * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_transposed_adj_list_starts_orig, h_transposed_adj_list_starts.data(), (V_original + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_transposed_adj_list_edges_orig, h_transposed_adj_list_edges.data(), E_original * sizeof(int), cudaMemcpyHostToDevice));

    // --- CUDA Kernel Launches ---
    int blocks_for_original_nodes = (V_original + 255) / 256;
    int threads_per_block = 256;

    // Initializations for trimming
    initializeArrayKernel<<<blocks_for_original_nodes, threads_per_block>>>(d_node_status, 0, V_original);
    initializeArrayKernel<<<blocks_for_original_nodes, threads_per_block>>>(d_scc_id_original, -1, V_original); // Use original SCC ID array for initial assignment
    CUDA_CHECK(cudaDeviceSynchronize());

    computeDegreesKernel<<<blocks_for_original_nodes, threads_per_block>>>(V_original, d_adj_list_starts_orig, d_transposed_adj_list_starts_orig,
                                                        d_out_degree, d_in_degree);
    CUDA_CHECK(cudaDeviceSynchronize());
    // std::cout << "Initial degrees computed." << std::endl;

    // --- Trimming Step ---
    int total_scc_count = 0;
    int newly_trimmed_nodes_host = 1;
    // std::cout << "Starting Trimming Step..." << std::endl;
    while (newly_trimmed_nodes_host > 0) {
        CUDA_CHECK(cudaMemset(d_trimmed_count_gpu, 0, sizeof(int)));
        newly_trimmed_nodes_host = 0;

        trimGraphKernel<<<blocks_for_original_nodes, threads_per_block>>>(
            V_original, d_adj_list_starts_orig, d_adj_list_edges_orig,
            d_transposed_adj_list_starts_orig, d_transposed_adj_list_edges_orig,
            d_out_degree, d_in_degree,
            d_node_status, d_scc_id_original, // Assign to original SCC ID array
            total_scc_count,
            d_trimmed_count_gpu
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&newly_trimmed_nodes_host, d_trimmed_count_gpu, sizeof(int), cudaMemcpyDeviceToHost));
        total_scc_count += newly_trimmed_nodes_host;
        // std::cout << "Trimmed " << newly_trimmed_nodes_host << " nodes in this iteration. Total trimmed: " << total_scc_count << std::endl;
    }
    int num_active_nodes = V_original - total_scc_count;
    // std::cout << "Trimming Step Complete. Remaining active nodes to process: " << num_active_nodes << std::endl;


    // --- Device Pointers for Compaction and Kosaraju's (initialized to nullptr) ---
    // These must be declared outside the if block because their cudaFree calls are outside.
    int* d_active_node_map = nullptr;
    int* d_reverse_node_map = nullptr;
    int* d_adj_list_starts_compact = nullptr;
    int* d_adj_list_edges_compact = nullptr;
    int* d_transposed_adj_list_starts_compact = nullptr;
    int* d_transposed_adj_list_edges_compact = nullptr;
    int* d_new_total_edges_gpu_counter = nullptr;

    unsigned long long* d_current_frontier_bitset = nullptr;
    unsigned long long* d_next_frontier_bitset = nullptr;
    unsigned long long* d_visited_bfs_state = nullptr;
    unsigned long long* d_global_visited_first_pass = nullptr; // Renamed for clarity
    unsigned long long* d_global_visited_second_pass = nullptr; // New allocation for second pass
    int* d_sparse_frontier = nullptr;
    int* d_sparse_next_frontier = nullptr;
    int* d_sparse_frontier_count_gpu = nullptr;
    int* d_dense_frontier_count_gpu = nullptr;
    int* d_finishing_order = nullptr;
    int* d_current_time_gpu = nullptr;
    int* d_scc_id_compacted = nullptr;

    // --- Hoisted declarations for new_total_edges and ull_bitset_size ---
    int new_total_edges = 0;
    int ull_bitset_size = 0;


    // --- Compaction Step and Kosaraju's Algorithm ---
    if (num_active_nodes > 0) {
        std::vector<int> h_active_node_map(V_original);
        std::vector<int> h_reverse_node_map(num_active_nodes);
        std::vector<int> h_node_status(V_original);

        // std::cout << "Starting Compaction Step..." << std::endl;

        CUDA_CHECK(cudaMalloc(&d_active_node_map, V_original * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_reverse_node_map, num_active_nodes * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(h_node_status.data(), d_node_status, V_original * sizeof(int), cudaMemcpyDeviceToHost));

        int current_compact_id = 0;
        for (int i = 0; i < V_original; ++i) {
            if (h_node_status[i] == 0) { // If active
                h_active_node_map[i] = current_compact_id;
                h_reverse_node_map[current_compact_id] = i;
                current_compact_id++;
            }
        }
        CUDA_CHECK(cudaMemcpy(d_active_node_map, h_active_node_map.data(), V_original * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_reverse_node_map, h_reverse_node_map.data(), num_active_nodes * sizeof(int), cudaMemcpyHostToDevice));

        for (int i = 0; i < V_original; ++i) {
            if (h_node_status[i] == 0) { // If active
                for (int neighbor : h_adj[i]) {
                    if (h_node_status[neighbor] == 0) { // If neighbor is also active
                        new_total_edges++;
                    }
                }
            }
        }

        CUDA_CHECK(cudaMalloc(&d_adj_list_starts_compact, (num_active_nodes + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_adj_list_edges_compact, new_total_edges * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_transposed_adj_list_starts_compact, (num_active_nodes + 1) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_transposed_adj_list_edges_compact, new_total_edges * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_new_total_edges_gpu_counter, sizeof(int)));

        CUDA_CHECK(cudaMemset(d_new_total_edges_gpu_counter, 0, sizeof(int)));
        compactGraphKernel<<<blocks_for_original_nodes, threads_per_block>>>(
            V_original, d_adj_list_starts_orig, d_adj_list_edges_orig,
            d_node_status, d_active_node_map,
            d_adj_list_starts_compact, d_adj_list_edges_compact,
            d_new_total_edges_gpu_counter
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&d_adj_list_starts_compact[num_active_nodes], d_new_total_edges_gpu_counter, sizeof(int), cudaMemcpyDeviceToDevice));


        CUDA_CHECK(cudaMemset(d_new_total_edges_gpu_counter, 0, sizeof(int)));
        compactGraphKernel<<<blocks_for_original_nodes, threads_per_block>>>(
            V_original, d_transposed_adj_list_starts_orig, d_transposed_adj_list_edges_orig,
            d_node_status, d_active_node_map,
            d_transposed_adj_list_starts_compact, d_transposed_adj_list_edges_compact,
            d_new_total_edges_gpu_counter
        );
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&d_transposed_adj_list_starts_compact[num_active_nodes], d_new_total_edges_gpu_counter, sizeof(int), cudaMemcpyDeviceToDevice));


        // std::cout << "Compaction Step Complete. New graph has " << num_active_nodes << " nodes and " << new_total_edges << " edges." << std::endl;


        // --- Kosaraju's Algorithm on Compacted Graph ---
        ull_bitset_size = get_ull_size(num_active_nodes);

        // Calculate blocks for kernels operating on num_active_nodes
        int blocks_for_active_nodes = (num_active_nodes + threads_per_block - 1) / threads_per_block;
        // Calculate blocks for kernels operating on ull_bitset_size
        int blocks_for_ull_bitset = (ull_bitset_size + threads_per_block - 1) / threads_per_block;


        // Common GPU memory for Kosaraju's passes
        CUDA_CHECK(cudaMalloc(&d_current_frontier_bitset, ull_bitset_size * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_next_frontier_bitset, ull_bitset_size * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_visited_bfs_state, ull_bitset_size * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMalloc(&d_global_visited_first_pass, ull_bitset_size * sizeof(unsigned long long))); // Distinct for pass 1
        CUDA_CHECK(cudaMalloc(&d_global_visited_second_pass, ull_bitset_size * sizeof(unsigned long long))); // Distinct for pass 2
        CUDA_CHECK(cudaMalloc(&d_sparse_frontier, num_active_nodes * sizeof(int))); // Max size for sparse frontier
        CUDA_CHECK(cudaMalloc(&d_sparse_next_frontier, num_active_nodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_sparse_frontier_count_gpu, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_dense_frontier_count_gpu, sizeof(int)));

        CUDA_CHECK(cudaMalloc(&d_finishing_order, num_active_nodes * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_current_time_gpu, sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_scc_id_compacted, num_active_nodes * sizeof(int)));


        // --- First Pass: Approximate Finishing Times ---
        initializeBitsetKernel<<<blocks_for_ull_bitset, threads_per_block>>>(d_global_visited_first_pass, ull_bitset_size);
        CUDA_CHECK(cudaMemset(d_current_time_gpu, 0, sizeof(int)));
        CUDA_CHECK(cudaDeviceSynchronize());
        // std::cout << "Starting First Pass (Kosaraju) on compacted graph..." << std::endl;

        std::vector<unsigned long long> h_global_visited_first_pass_vec(ull_bitset_size);

        for (int start_compact_node = 0; start_compact_node < num_active_nodes; ++start_compact_node) {
            // DEBUG PRINT: Check values before cudaMemcpy
            // std::cout << "DEBUG (Before Memcpy): start_compact_node=" << start_compact_node
            //           << ", ull_bitset_size=" << ull_bitset_size << std::endl;

            CUDA_CHECK(cudaMemcpy(h_global_visited_first_pass_vec.data(), d_global_visited_first_pass, ull_bitset_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

            if (!host_is_bit_set(h_global_visited_first_pass_vec, start_compact_node)) { // Check on host
                // Start new BFS
                initializeBitsetKernel<<<blocks_for_ull_bitset, threads_per_block>>>(d_visited_bfs_state, ull_bitset_size);
                initializeBitsetKernel<<<blocks_for_ull_bitset, threads_per_block>>>(d_current_frontier_bitset, ull_bitset_size);
                initializeBitsetKernel<<<blocks_for_ull_bitset, threads_per_block>>>(d_next_frontier_bitset, ull_bitset_size);

                setBitKernel<<<1,1>>>(d_current_frontier_bitset, start_compact_node);
                setBitKernel<<<1,1>>>(d_global_visited_first_pass, start_compact_node);
                setBitKernel<<<1,1>>>(d_visited_bfs_state, start_compact_node);
                CUDA_CHECK(cudaDeviceSynchronize()); // Ensure bits are set before proceeding

                int current_frontier_size = 1;

                // This is a subtle point for accurate Kosaraju BFS approximation
                // The starting node's 'finishing' order is recorded here.
                markFinishedKernel<<<blocks_for_active_nodes, threads_per_block>>>(num_active_nodes, d_finishing_order, d_current_time_gpu, d_current_frontier_bitset);
                CUDA_CHECK(cudaDeviceSynchronize());


                while (current_frontier_size > 0) {
                    CUDA_CHECK(cudaMemset(d_sparse_frontier_count_gpu, 0, sizeof(int)));
                    CUDA_CHECK(cudaMemset(d_dense_frontier_count_gpu, 0, sizeof(int)));
                    initializeBitsetKernel<<<blocks_for_ull_bitset, threads_per_block>>>(d_next_frontier_bitset, ull_bitset_size);

                    if (current_frontier_size <= SPARSE_THRESHOLD) { // Sparse mode
                        int* h_current_frontier_list = new int[current_frontier_size];
                        std::vector<unsigned long long> h_current_frontier_bitset_vec_temp(ull_bitset_size);
                        CUDA_CHECK(cudaMemcpy(h_current_frontier_bitset_vec_temp.data(), d_current_frontier_bitset, ull_bitset_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

                        int current_pos = 0;
                        for(int i = 0; i < num_active_nodes; ++i) { // Iterating up to num_active_nodes is fine for scanning the bitset
                            if (host_is_bit_set(h_current_frontier_bitset_vec_temp, i)) {
                                h_current_frontier_list[current_pos++] = i;
                            }
                        }
                        CUDA_CHECK(cudaMemcpy(d_sparse_frontier, h_current_frontier_list, current_frontier_size * sizeof(int), cudaMemcpyHostToDevice));
                        delete[] h_current_frontier_list; // Free host memory


                        bfs_sparse_kernel<<< (current_frontier_size + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
                            current_frontier_size, d_adj_list_starts_compact, d_adj_list_edges_compact,
                            d_visited_bfs_state, d_global_visited_first_pass,
                            d_sparse_frontier, d_sparse_next_frontier, d_sparse_frontier_count_gpu
                        );
                    } else { // Dense mode
                         bfs_dense_kernel<<<blocks_for_active_nodes, threads_per_block>>>( // Correct blocks here
                            num_active_nodes, d_adj_list_starts_compact, d_adj_list_edges_compact,
                            d_visited_bfs_state, d_global_visited_first_pass,
                            d_current_frontier_bitset, d_next_frontier_bitset, d_dense_frontier_count_gpu
                        );
                    }
                    CUDA_CHECK(cudaDeviceSynchronize());

                    // Swap frontiers
                    unsigned long long* temp_bitset = d_current_frontier_bitset;
                    d_current_frontier_bitset = d_next_frontier_bitset;
                    d_next_frontier_bitset = temp_bitset; // d_next_frontier_bitset now holds the previous 'current'

                    if (current_frontier_size <= SPARSE_THRESHOLD) {
                        CUDA_CHECK(cudaMemcpy(&current_frontier_size, d_sparse_frontier_count_gpu, sizeof(int), cudaMemcpyDeviceToHost));
                    } else {
                        CUDA_CHECK(cudaMemcpy(&current_frontier_size, d_dense_frontier_count_gpu, sizeof(int), cudaMemcpyDeviceToHost));
                    }

                    // Record finishing times for nodes that were in the just processed (now d_next_frontier_bitset)
                    // This call ensures nodes are marked finished as soon as their exploration is complete.
                    // It should be launched on the bitset that just completed its level.
                    markFinishedKernel<<<blocks_for_active_nodes, threads_per_block>>>(num_active_nodes,
                                                                 d_finishing_order, d_current_time_gpu,
                                                                 d_next_frontier_bitset); // This is the old d_current_frontier_bitset
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
            }
        }
        // std::cout << "First Pass Complete." << std::endl;

        // --- Second Pass: Identify SCCs on compacted graph ---
        std::vector<int> h_finishing_order(num_active_nodes);
        CUDA_CHECK(cudaMemcpy(h_finishing_order.data(), d_finishing_order, num_active_nodes * sizeof(int), cudaMemcpyDeviceToHost));

        // Initialize global visited for second pass
        initializeBitsetKernel<<<blocks_for_ull_bitset, threads_per_block>>>(d_global_visited_second_pass, ull_bitset_size);
        CUDA_CHECK(cudaDeviceSynchronize());
        // std::cout << "Starting Second Pass (Kosaraju) on compacted graph..." << std::endl;

        std::vector<unsigned long long> h_global_visited_second_pass_vec(ull_bitset_size);

        for (int i = 0; i < num_active_nodes; ++i) {
            int compact_node = h_finishing_order[i]; // Node from finishing order

            // DEBUG PRINT: Check values before cudaMemcpy
            // std::cout << "DEBUG (Before Memcpy Pass 2): compact_node=" << compact_node
            //           << ", ull_bitset_size=" << ull_bitset_size << std::endl;

            CUDA_CHECK(cudaMemcpy(h_global_visited_second_pass_vec.data(), d_global_visited_second_pass, ull_bitset_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

            if (!host_is_bit_set(h_global_visited_second_pass_vec, compact_node)) { // Check on host
                total_scc_count++; // New SCC found
                // Start new BFS on transposed graph
                initializeBitsetKernel<<<blocks_for_ull_bitset, threads_per_block>>>(d_visited_bfs_state, ull_bitset_size);
                initializeBitsetKernel<<<blocks_for_ull_bitset, threads_per_block>>>(d_current_frontier_bitset, ull_bitset_size);
                initializeBitsetKernel<<<blocks_for_ull_bitset, threads_per_block>>>(d_next_frontier_bitset, ull_bitset_size);

                setBitKernel<<<1,1>>>(d_current_frontier_bitset, compact_node);
                setBitKernel<<<1,1>>>(d_global_visited_second_pass, compact_node);
                setBitKernel<<<1,1>>>(d_visited_bfs_state, compact_node);
                // Assign SCC ID directly to the compacted node ID for now
                // We'll map it back to original IDs at the very end
                assignSccIdKernel<<<1,1>>>(d_scc_id_compacted, compact_node, total_scc_count);
                CUDA_CHECK(cudaDeviceSynchronize());

                int current_frontier_size = 1;

                while (current_frontier_size > 0) {
                    CUDA_CHECK(cudaMemset(d_sparse_frontier_count_gpu, 0, sizeof(int)));
                    CUDA_CHECK(cudaMemset(d_dense_frontier_count_gpu, 0, sizeof(int)));
                    initializeBitsetKernel<<<blocks_for_ull_bitset, threads_per_block>>>(d_next_frontier_bitset, ull_bitset_size);

                    if (current_frontier_size <= SPARSE_THRESHOLD) { // Sparse mode
                        int* h_current_frontier_list = new int[current_frontier_size];
                        std::vector<unsigned long long> h_current_frontier_bitset_vec_temp(ull_bitset_size);
                        CUDA_CHECK(cudaMemcpy(h_current_frontier_bitset_vec_temp.data(), d_current_frontier_bitset, ull_bitset_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
                        int current_pos = 0;
                        for(int j = 0; j < num_active_nodes; ++j) {
                            if (host_is_bit_set(h_current_frontier_bitset_vec_temp, j)) {
                                h_current_frontier_list[current_pos++] = j;
                            }
                        }
                        CUDA_CHECK(cudaMemcpy(d_sparse_frontier, h_current_frontier_list, current_frontier_size * sizeof(int), cudaMemcpyHostToDevice));
                        delete[] h_current_frontier_list;

                        scc_sparse_kernel<<< (current_frontier_size + threads_per_block - 1) / threads_per_block, threads_per_block>>>(
                            current_frontier_size, d_transposed_adj_list_starts_compact, d_transposed_adj_list_edges_compact,
                            d_visited_bfs_state, d_global_visited_second_pass,
                            d_scc_id_compacted, total_scc_count,
                            d_sparse_frontier, d_sparse_next_frontier, d_sparse_frontier_count_gpu
                        );
                    } else { // Dense mode
                        scc_dense_kernel<<<blocks_for_active_nodes, threads_per_block>>>( // Correct blocks here
                            num_active_nodes, d_transposed_adj_list_starts_compact, d_transposed_adj_list_edges_compact,
                            d_visited_bfs_state, d_global_visited_second_pass,
                            d_scc_id_compacted, total_scc_count,
                            d_current_frontier_bitset, d_next_frontier_bitset, d_dense_frontier_count_gpu
                        );
                    }
                    CUDA_CHECK(cudaDeviceSynchronize());

                    unsigned long long* temp_bitset = d_current_frontier_bitset;
                    d_current_frontier_bitset = d_next_frontier_bitset;
                    d_next_frontier_bitset = temp_bitset;

                    if (current_frontier_size <= SPARSE_THRESHOLD) {
                        CUDA_CHECK(cudaMemcpy(&current_frontier_size, d_sparse_frontier_count_gpu, sizeof(int), cudaMemcpyDeviceToHost));
                    } else {
                        CUDA_CHECK(cudaMemcpy(&current_frontier_size, d_dense_frontier_count_gpu, sizeof(int), cudaMemcpyDeviceToHost));
                    }
                }
            }
        }
        // std::cout << "Second Pass Complete." << std::endl;

        // --- Map compacted SCC IDs back to original node IDs ---
        mapSccIdsBackKernel<<<blocks_for_active_nodes, threads_per_block>>>( // Correct blocks here
            num_active_nodes, d_scc_id_compacted, d_reverse_node_map, d_scc_id_original
        );
        CUDA_CHECK(cudaDeviceSynchronize());
    } // End of if (num_active_nodes > 0) block


    // --- Final Cleanup and Output ---

    // std::cout << "Number of Strongly Connected Components: " << total_scc_count << std::endl;

    std::vector<int> h_scc_id_final(V_original);
    CUDA_CHECK(cudaMemcpy(h_scc_id_final.data(), d_scc_id_original, V_original * sizeof(int), cudaMemcpyDeviceToHost));

    // std::cout << "\nNode to SCC ID mapping (Original Node IDs):" << std::endl;
    for (int i = 0; i < V_original; ++i) {
        // std::cout << "Node " << i << ": SCC " << h_scc_id_final[i] << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double time_taken_seconds = std::chrono::duration<double>(end_time - start_time).count();

    // --- Performance Metrics (Approximation) ---
    double bytes_accessed = (2.0 * (V_original + 1) * sizeof(int)) + (2.0 * E_original * sizeof(int)) + // Original graph
                            (V_original * sizeof(int)) * 3 + // out_degree, in_degree, node_status
                            sizeof(int); // d_trimmed_count_gpu (approx)

    if (num_active_nodes > 0) { // Only count these if they were allocated/used
        bytes_accessed += (2.0 * (num_active_nodes + 1) * sizeof(int)) + (2.0 * new_total_edges * sizeof(int)); // Compacted graph
        bytes_accessed += (V_original * sizeof(int)) * 2; // active_node_map, reverse_node_map
        bytes_accessed += (long long)(ull_bitset_size) * sizeof(unsigned long long) * 4; // bitsets (4 separate full size: current_frontier, next_frontier, visited_bfs, global_visited_first_pass)
        bytes_accessed += (long long)(num_active_nodes) * sizeof(int) * 2; // sparse frontiers
        bytes_accessed += sizeof(int) * 4; // various counters (sparse/dense count, current time, new total edges)
        bytes_accessed += (long long)(num_active_nodes) * sizeof(int); // finishing order
        bytes_accessed += (long long)(num_active_nodes) * sizeof(int); // compacted scc ids
    }


    double gb_accessed = bytes_accessed / (1024.0 * 1024.0 * 1024.0);
    double throughput_gbps = gb_accessed / time_taken_seconds;

    // std::cout << "\nTotal Execution Time: " << time_taken_seconds * 1000 << " ms" << std::endl;
    std::cout << "Estimated Memory Access: " << gb_accessed << " GB" << std::endl;
    std::cout << "Estimated Memory Throughput: " << throughput_gbps << " GB/s" << std::endl;


    // --- Cleanup ---
    // These are always allocated, regardless of num_active_nodes
    CUDA_CHECK(cudaFree(d_adj_list_starts_orig));
    CUDA_CHECK(cudaFree(d_adj_list_edges_orig));
    CUDA_CHECK(cudaFree(d_transposed_adj_list_starts_orig));
    CUDA_CHECK(cudaFree(d_transposed_adj_list_edges_orig));
    CUDA_CHECK(cudaFree(d_out_degree));
    CUDA_CHECK(cudaFree(d_in_degree));
    CUDA_CHECK(cudaFree(d_node_status));
    CUDA_CHECK(cudaFree(d_scc_id_original));
    CUDA_CHECK(cudaFree(d_trimmed_count_gpu));

    if (num_active_nodes > 0) { // Only free if allocated
        CUDA_CHECK(cudaFree(d_active_node_map));
        CUDA_CHECK(cudaFree(d_reverse_node_map));
        CUDA_CHECK(cudaFree(d_adj_list_starts_compact));
        CUDA_CHECK(cudaFree(d_adj_list_edges_compact));
        CUDA_CHECK(cudaFree(d_transposed_adj_list_starts_compact));
        CUDA_CHECK(cudaFree(d_transposed_adj_list_edges_compact));
        CUDA_CHECK(cudaFree(d_new_total_edges_gpu_counter));

        CUDA_CHECK(cudaFree(d_current_frontier_bitset));
        CUDA_CHECK(cudaFree(d_next_frontier_bitset));
        CUDA_CHECK(cudaFree(d_visited_bfs_state));
        CUDA_CHECK(cudaFree(d_global_visited_first_pass));
        CUDA_CHECK(cudaFree(d_global_visited_second_pass)); // Free the new allocation
        CUDA_CHECK(cudaFree(d_sparse_frontier));
        CUDA_CHECK(cudaFree(d_sparse_next_frontier));
        CUDA_CHECK(cudaFree(d_sparse_frontier_count_gpu));
        CUDA_CHECK(cudaFree(d_dense_frontier_count_gpu));
        CUDA_CHECK(cudaFree(d_finishing_order));
  d_finishing_order = nullptr; // Defensive nulling
        CUDA_CHECK(cudaFree(d_current_time_gpu));
        CUDA_CHECK(cudaFree(d_scc_id_compacted));
    }

    return 0;
}