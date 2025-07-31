#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <unordered_set>
#include <utility>
#include <atomic>
#include <stack> // For iterative DFS in WCC
#include <set>   // For sorting and merging in WCC

using namespace std;

// Global atomic array to track processed nodes across all SCC tasks
// Initialized in printSCCs
atomic<bool>* global_is_processed = nullptr;
int global_num_nodes = 0;

// A fully parallel BFS that explores the graph level by level.
// Returns the set of reachable nodes.
// This BFS is now simpler, it doesn't need to consider 'is_node_processed' for other SCCs.
// It just finds nodes reachable from start_node within its own scope.
unordered_set<int> parallel_bfs(int start_node, const vector<vector<int>>& adj) {
    unordered_set<int> reachable_nodes;
    if (start_node < 0 || start_node >= adj.size()) {
        return reachable_nodes;
    }

    // Heuristic pre-allocation
    reachable_nodes.reserve(adj.size() / 4);

    vector<int> current_level_nodes;
    vector<int> next_level_nodes;
    next_level_nodes.reserve(adj.size() / 10); // Heuristic

    current_level_nodes.push_back(start_node);
    reachable_nodes.insert(start_node);

    while (!current_level_nodes.empty()) {
        next_level_nodes.clear(); // Clear for next level

        // Use thread-local vectors to collect next-level nodes without locking.
        vector<vector<int>> next_level_thread_local(omp_get_max_threads());

        #pragma omp parallel for default(none) shared(current_level_nodes, adj, next_level_thread_local, reachable_nodes)
        for (size_t i = 0; i < current_level_nodes.size(); ++i) {
            int u = current_level_nodes[i];
            int thread_id = omp_get_thread_num();
            for (int v : adj[u]) {
                // Check if already visited in this BFS. Using find is thread-safe for const access.
                // We will collect and then insert in a single critical section or merge after the parallel loop.
                if (reachable_nodes.find(v) == reachable_nodes.end()) {
                    next_level_thread_local[thread_id].push_back(v);
                }
            }
        }

        // Merge results from all threads into next_level_nodes and reachable_nodes.
        // This is done sequentially or with a single critical section for 'reachable_nodes' set insertion.
        for (const auto& vec : next_level_thread_local) {
            for (int node : vec) {
                // Only push to next_level_nodes if it's a *newly discovered* node for this BFS
                // and insert into reachable_nodes for tracking within this BFS.
                if (reachable_nodes.insert(node).second) {
                    next_level_nodes.push_back(node);
                }
            }
        }
        current_level_nodes.swap(next_level_nodes);
    }
    return reachable_nodes;
}

// Function to find WCCs using parallel BFS/DFS on the undirected graph
// Returns a vector of vectors, where each inner vector is a WCC
vector<vector<int>> find_wccs(int V, const vector<vector<int>>& adj) {
    vector<vector<int>> wccs_list;
    if (V == 0) return wccs_list;

    vector<bool> visited(V, false);
    // Create an undirected graph for WCC finding
    vector<vector<int>> undirected_adj(V);
    for (int u = 0; u < V; ++u) {
        for (int v : adj[u]) {
            undirected_adj[u].push_back(v);
            undirected_adj[v].push_back(u);
        }
    }

    #pragma omp parallel for default(none) shared(V, undirected_adj, visited, wccs_list) schedule(dynamic)
    for (int i = 0; i < V; ++i) {
        if (!visited[i]) {
            vector<int> current_wcc;
            stack<int> s;

            #pragma omp critical(visited_check)
            {
                // Double check if visited to avoid redundant work if another thread just processed it
                if (!visited[i]) {
                    s.push(i);
                    visited[i] = true;
                    current_wcc.push_back(i);
                }
            }

            // Perform a local BFS/DFS for the WCC
            if (!s.empty()) {
                vector<int> q_current_level;
                q_current_level.push_back(s.top());
                s.pop();

                while(!q_current_level.empty()){
                    vector<int> q_next_level;
                    for(int u : q_current_level){
                        for(int v : undirected_adj[u]){
                            if(!visited[v]){
                                #pragma omp critical(visited_mark)
                                {
                                    if(!visited[v]){ // Double check after lock
                                        visited[v] = true;
                                        current_wcc.push_back(v);
                                        q_next_level.push_back(v);
                                    }
                                }
                            }
                        }
                    }
                    q_current_level.swap(q_next_level);
                }

                #pragma omp critical(wccs_list_add)
                {
                    if (!current_wcc.empty()) {
                        wccs_list.push_back(std::move(current_wcc));
                    }
                }
            }
        }
    }
    return wccs_list;
}

// Main recursive function to find SCCs using parallel BFS
void parallel_scc_recursive(
    vector<int> nodes_in_component, // Nodes within the current sub-problem (e.g., a WCC or a remaining part)
    const vector<vector<int>>& adj,
    const vector<vector<int>>& adj_transpose,
    vector<vector<int>>& scc_list) {

    // Filter out already globally processed nodes from the current component's nodes
    vector<int> active_nodes;
    active_nodes.reserve(nodes_in_component.size());
    for(int node : nodes_in_component) {
        if (!global_is_processed[node].load(memory_order_relaxed)) {
            active_nodes.push_back(node);
        }
    }
    nodes_in_component = std::move(active_nodes);

    if (nodes_in_component.empty()) {
        return;
    }

    // Pivot selection: The first unprocessed node in the sub-component
    int pivot = nodes_in_component.front(); // After filtering, this is guaranteed to be unprocessed

    unordered_set<int> forward_reachable, backward_reachable;

    #pragma omp task shared(forward_reachable, adj)
    {
        forward_reachable = parallel_bfs(pivot, adj);
    }

    #pragma omp task shared(backward_reachable, adj_transpose)
    {
        backward_reachable = parallel_bfs(pivot, adj_transpose);
    }

    #pragma omp taskwait

    vector<int> current_scc;
    current_scc.reserve(min(forward_reachable.size(), backward_reachable.size()));
    // Efficient intersection: iterate over the smaller set
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

    // Mark nodes in the found SCC as processed globally
    for (int node : current_scc) {
        global_is_processed[node].store(true, memory_order_relaxed);
    }

    #pragma omp critical(scc_list_add)
    if (!current_scc.empty()) {
        scc_list.push_back(std::move(current_scc));
    }

    // Divide remaining nodes for recursive calls
    vector<int> fwd_remainder, bwd_remainder, other_remainder;
    fwd_remainder.reserve(nodes_in_component.size());
    bwd_remainder.reserve(nodes_in_component.size());
    other_remainder.reserve(nodes_in_component.size());

    for (int node : nodes_in_component) {
        if (global_is_processed[node].load(memory_order_relaxed)) {
            continue; // Skip nodes already part of an SCC
        }

        bool in_fwd = forward_reachable.count(node);
        bool in_bwd = backward_reachable.count(node);

        if (in_fwd && !in_bwd) {
            fwd_remainder.push_back(node);
        } else if (!in_fwd && in_bwd) {
            bwd_remainder.push_back(node);
        } else if (!in_fwd && !in_bwd) {
            other_remainder.push_back(node);
        }
    }

    // Create tasks for recursive calls
    #pragma omp task
    parallel_scc_recursive(std::move(fwd_remainder), adj, adj_transpose, scc_list);
    #pragma omp task
    parallel_scc_recursive(std::move(bwd_remainder), adj, adj_transpose, scc_list);
    #pragma omp task
    parallel_scc_recursive(std::move(other_remainder), adj, adj_transpose, scc_list);
}


void printSCCs(int V, const vector<vector<int>>& adj) {
    vector<vector<int>> scc_list;
    if (V == 0) return;

    // Initialize global_is_processed for thread-safe tracking
    global_num_nodes = V;
    global_is_processed = new atomic<bool>[V];
    for (int i = 0; i < V; ++i) {
        global_is_processed[i].store(false, memory_order_relaxed);
    }

    // Reserve memory for the final list to avoid reallocations.
    scc_list.reserve(V / 2); // Heuristic

    // --- Trimming Step 1: Trivial (size-1) SCCs ---
    // This part is kept sequential as its impact on overall performance is usually minimal
    // compared to the main recursive algorithm, and parallelizing it adds overhead.
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
            if (!global_is_processed[i].load(memory_order_relaxed)) {
                scc_list.push_back({i});
                global_is_processed[i].store(true, memory_order_relaxed);
            }
        }
    }

    // --- Trimming Step 2: Size-2 SCCs ---
    for (int u = 0; u < V; ++u) {
        if (global_is_processed[u].load(memory_order_relaxed)) continue;
        for (int v : adj[u]) {
            if (global_is_processed[v].load(memory_order_relaxed) || u >= v) continue;
            // Check for back-edge v -> u
            for (int w : adj[v]) {
                if (w == u) {
                    if (!global_is_processed[u].load(memory_order_relaxed) && !global_is_processed[v].load(memory_order_relaxed)) {
                        scc_list.push_back({u, v});
                        global_is_processed[u].store(true, memory_order_relaxed);
                        global_is_processed[v].store(true, memory_order_relaxed);
                    }
                    break;
                }
            }
        }
    }

    // --- Main Algorithm on Remaining Graph using WCC decomposition ---
    vector<vector<int>> adj_transpose(V);
    for (int u = 0; u < V; ++u) {
        for (int v : adj[u]) {
            adj_transpose[v].push_back(u);
        }
    }

    // Find WCCs on the undirected version of the graph
    vector<vector<int>> wccs = find_wccs(V, adj);

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (const auto& wcc_nodes : wccs) {
                // Filter out any nodes from this WCC that might have been processed by trimming
                vector<int> unprocessed_wcc_nodes;
                unprocessed_wcc_nodes.reserve(wcc_nodes.size());
                for(int node : wcc_nodes) {
                    if(!global_is_processed[node].load(memory_order_relaxed)) {
                        unprocessed_wcc_nodes.push_back(node);
                    }
                }

                if (!unprocessed_wcc_nodes.empty()) {
                    #pragma omp task
                    parallel_scc_recursive(std::move(unprocessed_wcc_nodes), adj, adj_transpose, scc_list);
                }
            }
        }
    }
    
    // Clean up global_is_processed
    delete[] global_is_processed;
    global_is_processed = nullptr;

    // --- Print final results ---
    cout << "Strongly Connected Components in the given graph:" << endl;
    int count = 0;
    // Sort SCCs by their first element for consistent output
    sort(scc_list.begin(), scc_list.end(), [](const vector<int>& a, const vector<int>& b) {
        if (a.empty() || b.empty()) return a.size() < b.size(); // Handle empty components if any
        return a[0] < b[0];
    });

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
    ifstream inputFile("Slashdot0811.txt");
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
        if (u >= 0 && u < V && v_node >= 0 && v_node < V) {
            adj[u].push_back(v_node);
        }
    }

    inputFile.close();

    double start_time = omp_get_wtime();
    printSCCs(V, adj);
    double end_time = omp_get_wtime();

    double time = end_time - start_time;
    cout << "\nExecution time: " << time << " seconds" << endl;

    double bytes_accessed = 4.0 * (2.0 * E + 2.0 * V); // Approximation in bytes
    double gb_accessed = bytes_accessed / (1024.0 * 1024.0 * 1024.0); // Convert to GB
    double throughput_gbps = gb_accessed / time;

    cout << "Estimated Memory Access: " << gb_accessed << " GB" << endl;
    cout << "Estimated Memory Throughput: " << throughput_gbps << " GB/s" << endl;


    return 0;
}