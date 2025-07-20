#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <queue> // Still useful for initial BFS approach or level tracking
#include <unordered_set>
#include <utility>
#include <atomic> // For safer processed node tracking

using namespace std;

// A fully parallel BFS that explores the graph level by level.
// Returns the set of reachable nodes.
unordered_set<int> parallel_bfs(int start_node, int initial_graph_size, const vector<vector<int>>& adj, const atomic<bool>* is_node_processed) {
    unordered_set<int> reachable_nodes;
    if (start_node < 0 || start_node >= adj.size() || (is_node_processed && is_node_processed[start_node].load(memory_order_relaxed))) {
        return reachable_nodes;
    }

    // Pre-allocate memory for the unordered_set to reduce rehashing.
    reachable_nodes.reserve(initial_graph_size / 10); // Heuristic initial size

    queue<int> q;
    q.push(start_node);
    reachable_nodes.insert(start_node);

    // Vector to store nodes for the next level, built in parallel and then merged
    vector<int> current_level_nodes;

    while (!q.empty()) {
        current_level_nodes.clear();
        int level_size = q.size();
        current_level_nodes.reserve(level_size);
        for (int i = 0; i < level_size; ++i) {
            current_level_nodes.push_back(q.front());
            q.pop();
        }

        // Use thread-local vectors to collect next-level nodes without locking within the inner loop.
        vector<vector<int>> next_level_thread_local(omp_get_max_threads());

        // #pragma omp parallel for default(none) shared(current_level_nodes, adj, next_level_thread_local, reachable_nodes, is_node_processed)
        for (int i = 0; i < level_size; ++i) {
            int u = current_level_nodes[i];
            int thread_id = omp_get_thread_num();
            for (int v : adj[u]) {
                // Optimization: Skip if already processed by the main algorithm
                if (is_node_processed && is_node_processed[v].load(memory_order_relaxed)) {
                    continue;
                }
                // Check if already visited in this BFS. Using find is thread-safe for const access.
                if (reachable_nodes.find(v) == reachable_nodes.end()) {
                    next_level_thread_local[thread_id].push_back(v);
                }
            }
        }

        // Merge results from all threads into the main queue and reachable_nodes set.
        // This part needs to be sequential or use a single critical section for set insertion.
        // Merging thread-local vectors first is generally more efficient than many small critical sections.
        for (const auto& vec : next_level_thread_local) {
            for (int node : vec) {
                // This critical section is still needed for inserting into the shared reachable_nodes.
                // However, it's outside the inner loop of the BFS, reducing contention.
                #pragma omp critical(reachable_insert)
                {
                    if (reachable_nodes.insert(node).second) { // Only push if newly inserted
                        q.push(node);
                    }
                }
            }
        }
    }
    return reachable_nodes;
}


void parallel_scc_recursive(
    vector<int> nodes,
    const vector<vector<int>>& adj,
    const vector<vector<int>>& adj_transpose,
    vector<vector<int>>& scc_list,
    atomic<bool>* is_node_processed) { // Pass atomic array

    if (nodes.empty()) {
        return;
    }

    // Find a pivot that hasn't been processed yet within this subset of nodes
    int pivot = -1;
    for (int node : nodes) {
        if (!is_node_processed[node].load(memory_order_relaxed)) {
            pivot = node;
            break;
        }
    }

    if (pivot == -1) { // All nodes in this subset are already processed
        return;
    }

    int initial_graph_size = adj.size();

    unordered_set<int> forward_reachable, backward_reachable;

    #pragma omp task shared(forward_reachable, initial_graph_size, adj, is_node_processed)
    {
        forward_reachable = parallel_bfs(pivot, initial_graph_size, adj, is_node_processed);
    }

    #pragma omp task shared(backward_reachable, initial_graph_size, adj_transpose, is_node_processed)
    {
        backward_reachable = parallel_bfs(pivot, initial_graph_size, adj_transpose, is_node_processed);
    }

    #pragma omp taskwait

    vector<int> current_scc;
    // Iterate over the smaller set for efficiency
    if (forward_reachable.size() < backward_reachable.size()) {
        for (int node : forward_reachable) {
            if (backward_reachable.count(node)) {
                current_scc.push_back(node);
            }
        }
    } else {
        for (int node : backward_reachable) {
            if (forward_reachable.count(node)) {
                current_scc.push_back(node);
            }
        }
    }

    // Mark nodes in the found SCC as processed
    for (int node : current_scc) {
        is_node_processed[node].store(true, memory_order_relaxed);
    }

    #pragma omp critical(scc_list_add)
    if (!current_scc.empty()) {
        scc_list.push_back(std::move(current_scc));
    }

    // Prepare for recursive calls. Filter out already processed nodes.
    vector<int> fwd_remainder, bwd_remainder, other_remainder;
    fwd_remainder.reserve(nodes.size());
    bwd_remainder.reserve(nodes.size());
    other_remainder.reserve(nodes.size());

    for (int node : nodes) {
        if (is_node_processed[node].load(memory_order_relaxed)) { // Skip if already processed
            continue;
        }

        bool in_fwd = forward_reachable.count(node);
        bool in_bwd = backward_reachable.count(node);

        if (in_fwd && !in_bwd) { // Only in forward
            fwd_remainder.push_back(node);
        } else if (!in_fwd && in_bwd) { // Only in backward
            bwd_remainder.push_back(node);
        } else if (!in_fwd && !in_bwd) { // Neither in forward nor backward
            other_remainder.push_back(node);
        }
        // Nodes in (in_fwd && in_bwd) are part of current_scc and thus already processed
    }

    #pragma omp task
    parallel_scc_recursive(std::move(fwd_remainder), adj, adj_transpose, scc_list, is_node_processed);
    #pragma omp task
    parallel_scc_recursive(std::move(bwd_remainder), adj, adj_transpose, scc_list, is_node_processed);
    #pragma omp task
    parallel_scc_recursive(std::move(other_remainder), adj, adj_transpose, scc_list, is_node_processed);
}


void printSCCs(int V, const vector<vector<int>>& adj) {
    vector<vector<int>> scc_list;
    if (V == 0) return;

    scc_list.reserve(V / 2); // Heuristic reserve

    // Use atomic<bool> for thread-safe tracking of processed nodes
    vector<atomic<bool>> is_processed(V);
    for (int i = 0; i < V; ++i) {
        is_processed[i].store(false, memory_order_relaxed);
    }

    // --- Trimming Step 1: Trivial (size-1) SCCs ---
    // Can be parallelized for larger graphs, but for simplicity and potentially
    // small number of such nodes, a sequential pass is often fine.
    vector<int> in_degree(V, 0);
    vector<int> out_degree(V, 0);
    for(int u = 0; u < V; ++u) {
        out_degree[u] = adj[u].size();
        for(int v : adj[u]) {
            in_degree[v]++;
        }
    }
    for (int i = 0; i < V; ++i) {
        if (out_degree[i] == 0 || in_degree[i] == 0) {
            if (!is_processed[i].load(memory_order_relaxed)) {
                scc_list.push_back({i});
                is_processed[i].store(true, memory_order_relaxed);
            }
        }
    }

    // --- Trimming Step 2: Size-2 SCCs ---
    for (int u = 0; u < V; ++u) {
        if (is_processed[u].load(memory_order_relaxed)) continue;
        for (int v : adj[u]) {
            if (is_processed[v].load(memory_order_relaxed) || u >= v) continue; // Avoid duplicates and self-loops
            // Check for back-edge v -> u
            for (int w : adj[v]) {
                if (w == u) {
                    if (!is_processed[u].load(memory_order_relaxed) && !is_processed[v].load(memory_order_relaxed)) {
                        scc_list.push_back({u, v});
                        is_processed[u].store(true, memory_order_relaxed);
                        is_processed[v].store(true, memory_order_relaxed);
                    }
                    break;
                }
            }
        }
    }

    // --- Main Algorithm on Remaining Graph ---
    vector<int> remaining_nodes_vec;
    remaining_nodes_vec.reserve(V); // Over-reserve, then shrink_to_fit if needed
    for(int i = 0; i < V; ++i) {
        if(!is_processed[i].load(memory_order_relaxed)) {
            remaining_nodes_vec.push_back(i);
        }
    }

    if (!remaining_nodes_vec.empty()) {
        vector<vector<int>> adj_transpose(V);
        for (int u = 0; u < V; ++u) {
            for (int v : adj[u]) {
                adj_transpose[v].push_back(u);
            }
        }

        #pragma omp parallel
        {
            #pragma omp single nowait // nowait to allow other threads to potentially start working
            parallel_scc_recursive(std::move(remaining_nodes_vec), adj, adj_transpose, scc_list, is_processed.data());
        }
    }

    // --- Print final results ---
    cout << "Strongly Connected Components in the given graph:" << endl;
    int count = 0;
    for (const auto& component : scc_list) {
        cout << "Strongly Connected Component " << ++count << ": ";
        vector<int> sorted_component = component;
        sort(sorted_component.begin(), sorted_component.end());
        for (int node : sorted_component) {
            cout << node << " ";
        }
        cout << endl;
    }
    cout << "\nTotal Strongly Connected Components: " << scc_list.size() << endl;
}


int main() {
    ifstream inputFile("Wiki-Vote.txt");
    if (!inputFile.is_open()) {
        cerr << "Error opening the file!" << endl;
        return 1;
    }

    int V, E;
    inputFile >> V >> E;

    vector<vector<int>> adj(V);
    for (int i = 0; i < E; ++i) {
        int u, v_node;
        inputFile >> u >> v_node;
        if (u >= 0 && u < V && v_node >= 0 && v_node < V) { // Added boundary checks
            adj[u].push_back(v_node);
        }
    }

    inputFile.close();

    double start_time = omp_get_wtime();
    printSCCs(V, adj);
    double end_time = omp_get_wtime();

    cout << "\nExecution time: " << (end_time - start_time) << " seconds" << endl;
    double time = end_time - start_time;
    double bytes_accessed = 4.0 * (2.0 * E + 2.0 * V); // Approximation in bytes
    double gb_accessed = bytes_accessed / (1024.0 * 1024.0 * 1024.0); // Convert to GB
    double throughput_gbps = gb_accessed / time;

    cout << "Estimated Memory Access: " << gb_accessed << " GB" << endl;
    cout << "Estimated Memory Throughput: " << throughput_gbps << " GB/s" << endl;

    return 0;
}