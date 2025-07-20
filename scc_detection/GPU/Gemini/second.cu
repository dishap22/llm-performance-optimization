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

// Kernel to compute initial in-degrees and out-degrees
__global__ void computeDegreesKernel(int num_nodes, const int* d_adj_list_starts, const int* d_adj_list_edges,
                                    const int* d_transposed_adj_list_starts, const int* d_transposed_adj_list_edges,
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
                                int* d_node_status, // 0: active, 1: trimmed, assigned SCC ID
                                int* d_scc_id,
                                int current_scc_id_base, // Base for new SCC IDs
                                int* d_trimmed_count // Atomic counter for newly trimmed nodes
                                ) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_idx < num_nodes) {
        // Only process active nodes (status 0)
        if (d_node_status[node_idx] == 0) {
            // Check if it's a source (out-degree 0) or a sink (in-degree 0)
            if (d_out_degree[node_idx] == 0 || d_in_degree[node_idx] == 0) {
                // Atomically assign SCC ID and mark as trimmed
                int old_status = atomicCAS(&d_node_status[node_idx], 0, 1); // Mark as trimmed (status 1)
                if (old_status == 0) { // If successfully marked as trimmed
                    atomicExch(&d_scc_id[node_idx], current_scc_id_base + atomicAdd(d_trimmed_count, 1));

                    // Decrement in-degree of neighbors for sources (out-degree 0)
                    if (d_out_degree[node_idx] == 0) { // It's a source node
                        int start_edge = d_transposed_adj_list_starts[node_idx];
                        int end_edge = d_transposed_adj_list_starts[node_idx + 1];
                        for (int i = start_edge; i < end_edge; ++i) {
                            int neighbor = d_transposed_adj_list_edges[i]; // Neighbor in original graph points to this source
                            if (d_node_status[neighbor] == 0) { // Only affect active neighbors
                                atomicSub(&d_out_degree[neighbor], 1); // Its out-degree decreases as this edge is "removed"
                            }
                        }
                    }

                    // Decrement out-degree of neighbors for sinks (in-degree 0)
                    if (d_in_degree[node_idx] == 0) { // It's a sink node
                        int start_edge = d_adj_list_starts[node_idx];
                        int end_edge = d_adj_list_starts[node_idx + 1];
                        for (int i = start_edge; i < end_edge; ++i) {
                            int neighbor = d_adj_list_edges[i]; // Neighbor in original graph is pointed to by this sink
                            if (d_node_status[neighbor] == 0) { // Only affect active neighbors
                                atomicSub(&d_in_degree[neighbor], 1); // Its in-degree decreases as this edge is "removed"
                            }
                        }
                    }
                }
            }
        }
    }
}


// Kernel for the first pass: BFS-like traversal to determine an approximate finishing order.
__global__ void firstPassKernel(int num_nodes, const int* d_adj_list_starts, const int* d_adj_list_edges,
                                int* d_visited, // 0: unvisited, 1: visiting, 2: finished
                                int* d_finishing_order, int* d_current_time,
                                bool* d_queue, bool* d_next_queue, int* d_active_count,
                                const int* d_node_status // To ignore trimmed nodes
                                ) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_idx < num_nodes) {
        if (d_node_status[node_idx] != 0) return; // Skip trimmed nodes

        if (d_queue[node_idx]) { // If this node is active in the current queue
            // Mark node as visiting (0 -> 1)
            atomicExch(&d_visited[node_idx], 1);

            // Process neighbors
            int start_edge = d_adj_list_starts[node_idx];
            int end_edge = d_adj_list_starts[node_idx + 1];

            for (int i = start_edge; i < end_edge; ++i) {
                int neighbor = d_adj_list_edges[i];
                if (d_node_status[neighbor] == 0) { // Only consider active neighbors
                    // If neighbor is unvisited (0), try to mark it as visiting (1) and add to next queue
                    if (atomicCAS(&d_visited[neighbor], 0, 1) == 0) {
                        d_next_queue[neighbor] = true; // Add to next level's queue
                        atomicAdd(d_active_count, 1); // Increment active count for next iteration
                    }
                }
            }

            // This node is now processed for this level, mark it "finished" (1 -> 2)
            if (atomicExch(&d_visited[node_idx], 2) == 1) {
                int time = atomicAdd(d_current_time, 1);
                d_finishing_order[num_nodes - 1 - time] = node_idx; // Store in reverse order for Kosaraju
            }
            d_queue[node_idx] = false; // Deactivate from current queue
        }
    }
}

// Kernel for the second pass: BFS-like traversal on the transposed graph to identify SCCs.
__global__ void secondPassKernel(int num_nodes, const int* d_transposed_adj_list_starts, const int* d_transposed_adj_list_edges,
                                 int* d_visited, // 0: unvisited, 1: visited
                                 int* d_scc_id, int current_scc_id,
                                 bool* d_queue, bool* d_next_queue, int* d_active_count,
                                 const int* d_node_status // To ignore trimmed nodes
                                ) {
    int node_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (node_idx < num_nodes) {
        if (d_node_status[node_idx] != 0) return; // Skip trimmed nodes

        if (d_queue[node_idx]) { // If this node is active in the current queue
            // Mark as visited and assign SCC ID
            atomicExch(&d_visited[node_idx], 1); // Mark as visited (1) for second pass
            atomicExch(&d_scc_id[node_idx], current_scc_id); // Assign SCC ID

            // Process neighbors in the transposed graph
            int start_edge = d_transposed_adj_list_starts[node_idx];
            int end_edge = d_transposed_adj_list_starts[node_idx + 1];

            for (int i = start_edge; i < end_edge; ++i) {
                int neighbor = d_transposed_adj_list_edges[i];
                if (d_node_status[neighbor] == 0) { // Only consider active neighbors
                    // If neighbor is unvisited (0), try to mark it as visited (1) and add to next queue
                    if (atomicCAS(&d_visited[neighbor], 0, 1) == 0) {
                        d_next_queue[neighbor] = true; // Add to next level's queue
                        atomicAdd(d_active_count, 1); // Increment active count for next iteration
                    }
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
    h_adj_list_starts[V] = current_edge_idx;

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
    h_transposed_adj_list_starts[V] = current_edge_idx;

    // --- Device Memory Allocation ---
    int* d_adj_list_starts;
    int* d_adj_list_edges;
    int* d_transposed_adj_list_starts;
    int* d_transposed_adj_list_edges;
    int* d_out_degree;         // Current out-degree for trimming
    int* d_in_degree;          // Current in-degree for trimming
    int* d_node_status;        // 0: active, 1: trimmed, assigned SCC ID
    int* d_visited;            // For Kosaraju's BFS passes (0: unvisited, 1: visiting, 2: finished/visited)
    int* d_finishing_order;    // Stores node indices in their finishing order (reverse order for Kosaraju)
    int* d_current_time;       // Global counter for finishing times
    int* d_scc_id;             // Stores SCC ID for each node
    bool* d_queue;             // For BFS-like traversal
    bool* d_next_queue;        // For BFS-like traversal
    int* d_active_count;       // For counting active nodes in queue / trimmed nodes

    CUDA_CHECK(cudaMalloc(&d_adj_list_starts, (V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_adj_list_edges, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_transposed_adj_list_starts, (V + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_transposed_adj_list_edges, E * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_degree, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_in_degree, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_node_status, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_visited, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_finishing_order, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_current_time, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_scc_id, V * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_queue, V * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_next_queue, V * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_active_count, sizeof(int)));

    // --- Data Transfer (Host to Device) ---
    CUDA_CHECK(cudaMemcpy(d_adj_list_starts, h_adj_list_starts.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_adj_list_edges, h_adj_list_edges.data(), E * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_transposed_adj_list_starts, h_transposed_adj_list_starts.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_transposed_adj_list_edges, h_transposed_adj_list_edges.data(), E * sizeof(int), cudaMemcpyHostToDevice));

    // --- CUDA Kernel Launches ---

    int blocks = (V + 255) / 256;
    int threads_per_block = 256;

    // Initializations
    initializeArrayKernel<<<blocks, threads_per_block>>>(d_node_status, 0, V); // All nodes active initially
    initializeArrayKernel<<<blocks, threads_per_block>>>(d_scc_id, -1, V);     // No SCC ID assigned yet
    CUDA_CHECK(cudaDeviceSynchronize());

    // Compute initial degrees
    computeDegreesKernel<<<blocks, threads_per_block>>>(V, d_adj_list_starts, d_adj_list_edges,
                                                        d_transposed_adj_list_starts, d_transposed_adj_list_edges,
                                                        d_out_degree, d_in_degree);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Initial degrees computed." << std::endl;

    // --- Trimming Step ---
    int total_scc_count = 0;
    int newly_trimmed_nodes_host = 1; // Start with 1 to enter the loop
    std::cout << "Starting Trimming Step..." << std::endl;
    while (newly_trimmed_nodes_host > 0) {
        CUDA_CHECK(cudaMemset(d_active_count, 0, sizeof(int))); // Use d_active_count to track newly trimmed nodes
        newly_trimmed_nodes_host = 0;

        trimGraphKernel<<<blocks, threads_per_block>>>(
            V, d_adj_list_starts, d_adj_list_edges,
            d_transposed_adj_list_starts, d_transposed_adj_list_edges,
            d_out_degree, d_in_degree,
            d_node_status, d_scc_id,
            total_scc_count, // Base for new SCC IDs from trimming
            d_active_count
        );
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(&newly_trimmed_nodes_host, d_active_count, sizeof(int), cudaMemcpyDeviceToHost));
        total_scc_count += newly_trimmed_nodes_host;
        std::cout << "Trimmed " << newly_trimmed_nodes_host << " nodes in this iteration. Total trimmed: " << total_scc_count << std::endl;
    }
    std::cout << "Trimming Step Complete. Remaining active nodes to process: " << V - total_scc_count << std::endl;

    // --- Kosaraju's Algorithm on Remaining Graph ---
    // Initialize d_visited to 0, d_current_time to 0 for Kosaraju's part
    initializeArrayKernel<<<blocks, threads_per_block>>>(d_visited, 0, V);
    CUDA_CHECK(cudaMemset(d_current_time, 0, sizeof(int)));
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- First Pass: Approximate Finishing Times on active nodes ---
    std::cout << "Starting First Pass (Kosaraju) on remaining graph..." << std::endl;
    for (int start_node = 0; start_node < V; ++start_node) {
        if (d_node_status[start_node] != 0) continue; // Skip trimmed nodes

        int visited_status;
        CUDA_CHECK(cudaMemcpy(&visited_status, d_visited + start_node, sizeof(int), cudaMemcpyDeviceToHost));
        if (visited_status == 0) { // If node not yet visited and is active
            initializeBoolArrayKernel<<<blocks, threads_per_block>>>(d_queue, false, V);
            initializeBoolArrayKernel<<<blocks, threads_per_block>>>(d_next_queue, false, V);
            CUDA_CHECK(cudaMemset(d_active_count, 0, sizeof(int)));

            bool true_val = true;
            CUDA_CHECK(cudaMemcpy(d_queue + start_node, &true_val, sizeof(bool), cudaMemcpyHostToDevice));
            int initial_active = 1;
            CUDA_CHECK(cudaMemcpy(d_active_count, &initial_active, sizeof(int), cudaMemcpyHostToDevice));

            int current_active_nodes = 1;
            while (current_active_nodes > 0) {
                CUDA_CHECK(cudaMemset(d_active_count, 0, sizeof(int)));

                firstPassKernel<<<blocks, threads_per_block>>>(
                    V, d_adj_list_starts, d_adj_list_edges,
                    d_visited, d_finishing_order, d_current_time,
                    d_queue, d_next_queue, d_active_count, d_node_status
                );
                CUDA_CHECK(cudaDeviceSynchronize());

                bool* temp_queue = d_queue;
                d_queue = d_next_queue;
                d_next_queue = temp_queue;

                CUDA_CHECK(cudaMemcpy(&current_active_nodes, d_active_count, sizeof(int), cudaMemcpyDeviceToHost));
            }
        }
    }
    std::cout << "First Pass Complete." << std::endl;

    // --- Second Pass: Identify SCCs on active nodes ---
    std::vector<int> h_finishing_order(V);
    CUDA_CHECK(cudaMemcpy(h_finishing_order.data(), d_finishing_order, V * sizeof(int), cudaMemcpyDeviceToHost));

    // Reinitialize visited array for the second pass
    initializeArrayKernel<<<blocks, threads_per_block>>>(d_visited, 0, V);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "Starting Second Pass (Kosaraju) on remaining graph..." << std::endl;

    for (int i = 0; i < V; ++i) {
        int node = h_finishing_order[i]; // Get node from ordered list
        if (d_node_status[node] != 0) continue; // Skip trimmed nodes

        int visited_status;
        CUDA_CHECK(cudaMemcpy(&visited_status, d_visited + node, sizeof(int), cudaMemcpyDeviceToHost));

        if (visited_status == 0) { // If node not yet visited in this pass and is active
            total_scc_count++; // Increment overall SCC count for non-trivial SCCs
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
                    d_visited, d_scc_id, total_scc_count, // Pass current SCC ID
                    d_queue, d_next_queue, d_active_count, d_node_status
                );
                CUDA_CHECK(cudaDeviceSynchronize());

                bool* temp_queue = d_queue;
                d_queue = d_next_queue;
                d_next_queue = temp_queue;

                CUDA_CHECK(cudaMemcpy(&current_active_nodes, d_active_count, sizeof(int), cudaMemcpyDeviceToHost));
            }
        }
    }
    std::cout << "Second Pass Complete." << std::endl;

    std::cout << "Number of Strongly Connected Components: " << total_scc_count << std::endl;

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
    double bytes_accessed = (2.0 * (V + 1) * sizeof(int)) + (2.0 * E * sizeof(int)) +
                            (V * sizeof(int)) + (V * sizeof(int)) + (V * sizeof(int)) +
                            (V * sizeof(int)) + (V * sizeof(int)) + (V * sizeof(int)) + // 3 new for degrees/status
                            (2.0 * V * sizeof(bool)) + (2 * sizeof(int));

    double gb_accessed = bytes_accessed / (1024.0 * 1024.0 * 1024.0);
    double throughput_gbps = gb_accessed / time_taken_seconds;

    std::cout << "\nTotal Execution Time: " << time_taken_seconds * 1000 << " ms" << std::endl;
    std::cout << "Estimated Memory Access: " << gb_accessed << " GB" << std::endl;
    std::cout << "Estimated Memory Throughput: " << throughput_gbps << " GB/s" << std::endl;

    // --- Cleanup ---
    CUDA_CHECK(cudaFree(d_adj_list_starts));
    CUDA_CHECK(cudaFree(d_adj_list_edges));
    CUDA_CHECK(cudaFree(d_transposed_adj_list_starts));
    CUDA_CHECK(cudaFree(d_transposed_adj_list_edges));
    CUDA_CHECK(cudaFree(d_out_degree));
    CUDA_CHECK(cudaFree(d_in_degree));
    CUDA_CHECK(cudaFree(d_node_status));
    CUDA_CHECK(cudaFree(d_visited));
    CUDA_CHECK(cudaFree(d_finishing_order));
    CUDA_CHECK(cudaFree(d_current_time));
    CUDA_CHECK(cudaFree(d_scc_id));
    CUDA_CHECK(cudaFree(d_queue));
    CUDA_CHECK(cudaFree(d_next_queue));
    CUDA_CHECK(cudaFree(d_active_count));

    return 0;
}