#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <algorithm>
#include <chrono> // For measuring execution time
#include <numeric> // For std::iota

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

// --- Device Kernels ---

// Kernel to initialize an array on the device
__global__ void initializeArrayKernel(int* d_array, int value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = value;
    }
}

// Kernel to initialize a boolean array on the device
__global__ void initializeBoolArrayKernel(bool* d_array, bool value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] = value;
    }
}

// Kernel for the first pass: BFS-like traversal to determine an approximate finishing order.
// This kernel processes one "level" of the BFS.
__global__ void firstPassKernel(int num_nodes, const int* d_adj_list_starts, const int* d_adj_list_edges,
                                int* d_visited, int* d_finishing_order, int* d_current_time,
                                bool* d_queue, bool* d_next_queue, int* d_active_count) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_idx < num_nodes) {
        if (d_queue[node_idx]) { // If this node is active in the current queue
            // Mark node as visiting/visited (from 0 to 1)
            // This is effectively "visiting" it for this BFS level
            atomicExch(&d_visited[node_idx], 1);

            // Process neighbors
            int start_edge = d_adj_list_starts[node_idx];
            int end_edge = d_adj_list_starts[node_idx + 1];

            for (int i = start_edge; i < end_edge; ++i) {
                int neighbor = d_adj_list_edges[i];

                // If neighbor is unvisited (0), try to mark it as visited (1) and add to next queue
                if (atomicCAS(&d_visited[neighbor], 0, 1) == 0) {
                    d_next_queue[neighbor] = true; // Add to next level's queue
                    atomicAdd(d_active_count, 1); // Increment active count for next iteration
                }
            }

            // This node is now processed for this level, mark it "finished" for the first pass
            // and add its "finishing time" (order)
            // Use atomicExch to ensure only one thread marks it finished
            if (atomicExch(&d_visited[node_idx], 2) == 1) { // If it was '1' (visiting) and now finished (2)
                int time = atomicAdd(d_current_time, 1);
                d_finishing_order[num_nodes - 1 - time] = node_idx; // Store in reverse order for Kosaraju
            }
            d_queue[node_idx] = false; // Deactivate from current queue
        }
    }
}

// Kernel for the second pass: BFS-like traversal on the transposed graph to identify SCCs.
__global__ void secondPassKernel(int num_nodes, const int* d_transposed_adj_list_starts, const int* d_transposed_adj_list_edges,
                                 int* d_visited, int* d_scc_id, int current_scc_id,
                                 bool* d_queue, bool* d_next_queue, int* d_active_count) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_idx < num_nodes) {
        if (d_queue[node_idx]) { // If this node is active in the current queue
            // Mark as visited and assign SCC ID
            atomicExch(&d_visited[node_idx], 1); // Mark as visited (1) for second pass
            atomicExch(&d_scc_id[node_idx], current_scc_id); // Assign SCC ID

            // Process neighbors in the transposed graph
            int start_edge = d_transposed_adj_list_starts[node_idx];
            int end_edge = d_transposed_adj_list_starts[node_idx + 1];

            for (int i = start_edge; i < end_edge; ++i) {
                int neighbor = d_transposed_adj_list_edges[i];
                // If neighbor is unvisited (0), try to mark it as visited (1) and add to next queue
                if (atomicCAS(&d_visited[neighbor], 0, 1) == 0) {
                    d_next_queue[neighbor] = true; // Add to next level's queue
                    atomicAdd(d_active_count, 1); // Increment active count for next iteration
                }
            }
            d_queue[node_idx] = false; // Deactivate from current queue
        }
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

    int V, E;
    inputFile >> V >> E;

    // Host-side adjacency list representation for building CSR
    std::vector<std::vector<int>> h_adj(V);
    std::vector<std::vector<int>> h_adj_transposed(V); // For building transpose

    int u, v_node;
    for (int i = 0; i < E; ++i) {
        inputFile >> u >> v_node;
        h_adj[u].push_back(v_node);
        h_adj_transposed[v_node].push_back(u); // Building transpose directly
    }
    inputFile.close();

    // Convert adjacency list to CSR (Compressed Sparse Row) format for GPU
    // Original graph
    std::vector<int> h_adj_list_starts(V + 1, 0);
    std::vector<int> h_adj_list_edges(E);
    int current_edge_idx = 0;
    for (int i = 0; i < V; ++i) {
        h_adj_list_starts[i] = current_edge_idx;
        for (int neighbor : h_adj[i]) {
            h_adj_list_edges[current_edge_idx++] = neighbor;
        }
    }
    h_adj_list_starts[V] = current_edge_idx; // Last element points to total number of edges

    // Transposed graph
    std::vector<int> h_transposed_adj_list_starts(V + 1, 0);
    std::vector<int> h_transposed_adj_list_edges(E);
    current_edge_idx = 0;
    for (int i = 0; i < V; ++i) {
        h_transposed_adj_list_starts[i] = current_edge_idx;
        for (int neighbor : h_adj_transposed[i]) {
            h_transposed_adj_list_edges[current_edge_idx++] = neighbor;
        }
    }
    h_transposed_adj_list_starts[V] = current_edge_idx; // Last element points to total number of edges

    // --- Device Memory Allocation ---
    int* d_adj_list_starts;
    int* d_adj_list_edges;
    int* d_transposed_adj_list_starts;
    int* d_transposed_adj_list_edges;
    int* d_visited;            // 0: unvisited, 1: visiting (in current BFS), 2: visited/finished
    int* d_finishing_order;    // Stores node indices in their finishing order (reverse order for Kosaraju)
    int* d_current_time;       // Global counter for finishing times
    int* d_scc_id;             // Stores SCC ID for each node
    bool* d_queue;             // For BFS-like traversal
    bool* d_next_queue;        // For BFS-like traversal
    int* d_active_count;       // For counting active nodes in queue (to know when BFS level is done)

    CUDA_CHECK(cudaMalloc(&d_adj_list_starts, (V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_adj_list_edges, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_transposed_adj_list_starts, (V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_transposed_adj_list_edges, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_visited, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_finishing_order, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_current_time, sizeof(int))); // Single int for global time
    CUDA_CHECK(cudaMalloc(&d_scc_id, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_queue, V * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_next_queue, V * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_active_count, sizeof(int))); // For active nodes in queue

    // --- Data Transfer (Host to Device) ---
    CUDA_CHECK(cudaMemcpy(d_adj_list_starts, h_adj_list_starts.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adj_list_edges, h_adj_list_edges.data(), E * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_transposed_adj_list_starts, h_transposed_adj_list_starts.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_transposed_adj_list_edges, h_transposed_adj_list_edges.data(), E * sizeof(int), cudaMemcpyHostToDevice));

    // --- CUDA Kernel Launches ---

    int blocks = (V + 255) / 256; // Example block size
    int threads_per_block = 256;

    // Initialize d_visited to 0, d_scc_id to -1, d_current_time to 0
    initializeArrayKernel<<<blocks, threads_per_block>>>(d_visited, 0, V);
    initializeArrayKernel<<<blocks, threads_per_block>>>(d_scc_id, -1, V);
    CUDA_CHECK(cudaMemset(d_current_time, 0, sizeof(int))); // Set global time to 0
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure initializations are complete

    // --- First Pass: Approximate Finishing Times ---
    // Iterate through all nodes. If an unvisited node is found, start a new BFS-like traversal from it.
    std::cout << "Starting First Pass..." << std::endl;
    for (int start_node = 0; start_node < V; ++start_node) {
        int visited_status;
        CUDA_CHECK(cudaMemcpy(&visited_status, d_visited + start_node, sizeof(int), cudaMemcpyDeviceToHost));
        if (visited_status == 0) { // If node not yet visited, start a new traversal
            // Initialize queues for this BFS component
            initializeBoolArrayKernel<<<blocks, threads_per_block>>>(d_queue, false, V);
            initializeBoolArrayKernel<<<blocks, threads_per_block>>>(d_next_queue, false, V);
            CUDA_CHECK(cudaMemset(d_active_count, 0, sizeof(int))); // Reset active count for this BFS

            // Set the start_node in the current queue
            bool true_val = true;
            CUDA_CHECK(cudaMemcpy(d_queue + start_node, &true_val, sizeof(bool), cudaMemcpyHostToDevice));
            int initial_active = 1;
            CUDA_CHECK(cudaMemcpy(d_active_count, &initial_active, sizeof(int), cudaMemcpyHostToDevice)); // Set initial active count to 1

            int current_active_nodes = 1;
            while (current_active_nodes > 0) {
                // Reset d_active_count for the current kernel launch
                CUDA_CHECK(cudaMemset(d_active_count, 0, sizeof(int)));

                firstPassKernel<<<blocks, threads_per_block>>>(
                    V, d_adj_list_starts, d_adj_list_edges,
                    d_visited, d_finishing_order, d_current_time,
                    d_queue, d_next_queue, d_active_count
                );
                CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish

                // Swap queues for next iteration
                bool* temp_queue = d_queue;
                d_queue = d_next_queue;
                d_next_queue = temp_queue;

                // Get the count of active nodes for the next iteration
                CUDA_CHECK(cudaMemcpy(&current_active_nodes, d_active_count, sizeof(int), cudaMemcpyDeviceToHost));
            }
        }
    }
    std::cout << "First Pass Complete." << std::endl;

    // --- Second Pass: Identify SCCs ---
    // The d_finishing_order stores nodes in the order they finished.
    // For Kosaraju's, we need to process them in reverse order of finishing times.
    // Our firstPassKernel stores them in the correct order for Kosaraju, i.e.,
    // the node that finished last is at index 0, second last at index 1, and so on.
    // So we iterate d_finishing_order from index 0 to V-1.
    std::vector<int> h_finishing_order(V);
    CUDA_CHECK(cudaMemcpy(h_finishing_order.data(), d_finishing_order, V * sizeof(int), cudaMemcpyDeviceToHost));

    int scc_count = 0;
    // Reinitialize visited array for the second pass
    initializeArrayKernel<<<blocks, threads_per_block>>>(d_visited, 0, V); // 0: unvisited for this pass
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Starting Second Pass..." << std::endl;

    for (int i = 0; i < V; ++i) {
        int node = h_finishing_order[i]; // Get node from ordered list
        int visited_status;
        CUDA_CHECK(cudaMemcpy(&visited_status, d_visited + node, sizeof(int), cudaMemcpyDeviceToHost));

        if (visited_status == 0) { // If node not yet visited in this pass, it's the start of a new SCC
            scc_count++;
            // Start a new BFS-like traversal on the transposed graph
            initializeBoolArrayKernel<<<blocks, threads_per_block>>>(d_queue, false, V);
            initializeBoolArrayKernel<<<blocks, threads_per_block>>>(d_next_queue, false, V);
            CUDA_CHECK(cudaMemset(d_active_count, 0, sizeof(int)));

            bool true_val = true;
            CUDA_CHECK(cudaMemcpy(d_queue + node, &true_val, sizeof(bool), cudaMemcpyHostToDevice));
            int initial_active = 1;
            CUDA_CHECK(cudaMemcpy(d_active_count, &initial_active, sizeof(int), cudaMemcpyHostToDevice));

            int current_active_nodes = 1;
            while (current_active_nodes > 0) {
                CUDA_CHECK(cudaMemset(d_active_count, 0, sizeof(int)));

                secondPassKernel<<<blocks, threads_per_block>>>(
                    V, d_transposed_adj_list_starts, d_transposed_adj_list_edges,
                    d_visited, d_scc_id, scc_count, // Pass current SCC ID
                    d_queue, d_next_queue, d_active_count
                );
                CUDA_CHECK(cudaDeviceSynchronize());

                // Swap queues
                bool* temp_queue = d_queue;
                d_queue = d_next_queue;
                d_next_queue = temp_queue;

                // Get the count of active nodes for the next iteration
                CUDA_CHECK(cudaMemcpy(&current_active_nodes, d_active_count, sizeof(int), cudaMemcpyDeviceToHost));
            }
        }
    }
    std::cout << "Second Pass Complete." << std::endl;

    std::cout << "Number of Strongly Connected Components: " << scc_count << std::endl;

    // Optionally, print SCCs (copy d_scc_id back to host)
    std::vector<int> h_scc_id(V);
    CUDA_CHECK(cudaMemcpy(h_scc_id.data(), d_scc_id, V * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "\nNode to SCC ID mapping:" << std::endl;
    for (int i = 0; i < V; ++i) {
        std::cout << "Node " << i << ": SCC " << h_scc_id[i] << std::endl;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    double time_taken_seconds = std::chrono::duration<double>(end_time - start_time).count();

    // --- Performance Metrics (Approximation) ---
    // Memory accessed:
    // 2x (V+1) for starts array (original + transposed)
    // 2x E for edges array (original + transposed)
    // V for visited array
    // V for finishing_order array
    // V for scc_id array
    // 2x V for queue arrays
    // A few ints for counters.
    // All in bytes
    double bytes_accessed = (2.0 * (V + 1) * sizeof(int)) + (2.0 * E * sizeof(int)) +
                            (V * sizeof(int)) + (V * sizeof(int)) + (V * sizeof(int)) +
                            (2.0 * V * sizeof(bool)) + (2 * sizeof(int));

    double gb_accessed = bytes_accessed / (1024.0 * 1024.0 * 1024.0); // Convert to GB
    double throughput_gbps = gb_accessed / time_taken_seconds;

    std::cout << "\nTotal Execution Time: " << time_taken_seconds * 1000 << " ms" << std::endl;
    std::cout << "Estimated Memory Access: " << gb_accessed << " GB" << std::endl;
    std::cout << "Estimated Memory Throughput: " << throughput_gbps << " GB/s" << std::endl;

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_adj_list_starts));
    CUDA_CHECK(cudaFree(d_adj_list_edges));
    CUDA_CHECK(cudaFree(d_transposed_adj_list_starts));
    CUDA_CHECK(cudaFree(d_transposed_adj_list_edges));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_finishing_order));
    CUDA_CHECK(cudaFree(d_current_time));
    CUDA_CHECK(cudaFree(d_scc_id));
    CUDA_CHECK(cudaFree(d_queue));
    CUDA_CHECK(cudaFree(d_next_queue));
    CUDA_CHECK(cudaFree(d_active_count));

    return 0;
}