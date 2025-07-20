#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <omp.h>
#include <queue>
#include <unordered_set>
#include <utility> // For std::move
#include <atomic>
#include <chrono> // For timing

using namespace std;

// Performs a standard BFS and returns the set of reachable nodes by value.
// This is a safer pattern for parallelism than modifying a shared reference.
unordered_set<int> bfs(int start_node, const vector<vector<int>>& adj) {
    unordered_set<int> reachable_nodes;
    queue<int> q;

    if (start_node < 0 || start_node >= adj.size()) {
        return reachable_nodes; // Return empty set if start_node is invalid
    }

    q.push(start_node);
    reachable_nodes.insert(start_node);

    while (!q.empty()) {
        int u = q.front();
        q.pop();

        for (int v : adj[u]) {
            if (reachable_nodes.find(v) == reachable_nodes.end()) {
                reachable_nodes.insert(v);
                q.push(v);
            }
        }
    }
    return reachable_nodes;
}

// Recursively finds SCCs using the "divide and conquer" forward-backward approach.
void parallel_scc_recursive(
    vector<int> nodes, // Passed by value, moved into for efficiency
    const vector<vector<int>>& adj,
    const vector<vector<int>>& adj_transpose,
    vector<vector<int>>& scc_list) {
    
    if (nodes.empty()) {
        return;
    }

    int pivot = nodes.front();

    // The BFS tasks now return sets by value. We capture them into shared variables.
    unordered_set<int> forward_reachable, backward_reachable;
    
    #pragma omp task shared(forward_reachable)
    {
        forward_reachable = bfs(pivot, adj);
    }
    
    #pragma omp task shared(backward_reachable)
    {
        backward_reachable = bfs(pivot, adj_transpose);
    }

    #pragma omp taskwait

    vector<int> current_scc;
    current_scc.reserve(min(forward_reachable.size(), backward_reachable.size()));
    for (int node : forward_reachable) {
        if (backward_reachable.count(node)) {
            current_scc.push_back(node);
        }
    }
    
    // Writes to the final list must be synchronized.
    #pragma omp critical
    if (!current_scc.empty()) {
        scc_list.push_back(std::move(current_scc));
    }
    
    vector<int> fwd_remainder, bwd_remainder, other_remainder;
    unordered_set<int> scc_set(make_move_iterator(scc_list.back().begin()), make_move_iterator(scc_list.back().end()));
    if(scc_list.back().empty()) scc_set.clear();


    for (int node : nodes) {
        if (scc_set.count(node)) continue;
        
        bool in_fwd = forward_reachable.count(node);
        bool in_bwd = backward_reachable.count(node);
        
        if (in_fwd) fwd_remainder.push_back(node);
        else if (in_bwd) bwd_remainder.push_back(node);
        else other_remainder.push_back(node);
    }

    // Recurse on the remaining partitions as parallel tasks.
    #pragma omp task
    parallel_scc_recursive(std::move(fwd_remainder), adj, adj_transpose, scc_list);
    #pragma omp task
    parallel_scc_recursive(std::move(bwd_remainder), adj, adj_transpose, scc_list);
    #pragma omp task
    parallel_scc_recursive(std::move(other_remainder), adj, adj_transpose, scc_list);
}


// Main function to orchestrate the SCC finding process.
void printSCCs(int V, const vector<vector<int>>& adj) {
    vector<vector<int>> scc_list;
    vector<int> remaining_nodes_vec;

    if (V > 0) {
        vector<int> in_degree(V, 0);
        vector<int> out_degree(V, 0);
        
        #pragma omp parallel for
        for (int u = 0; u < V; ++u) {
            out_degree[u] = adj[u].size();
            for (int v : adj[u]) {
                #pragma omp atomic
                in_degree[v]++;
            }
        }

        for (int i = 0; i < V; ++i) {
            if (in_degree[i] == 0 || out_degree[i] == 0) {
                scc_list.push_back({i});
            } else {
                remaining_nodes_vec.push_back(i);
            }
        }
    }

    if (!remaining_nodes_vec.empty()) {
        vector<vector<int>> adj_transpose(V);
        #pragma omp parallel for
        for (int u = 0; u < V; ++u) {
            if (!adj[u].empty()){
                for (int v : adj[u]) {
                    #pragma omp critical
                    adj_transpose[v].push_back(u);
                }
            }
        }
        
        #pragma omp parallel
        {
            #pragma omp single
            parallel_scc_recursive(std::move(remaining_nodes_vec), adj, adj_transpose, scc_list);
        }
    }

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
    ifstream inputFile("twitter_combined.txt");
    if (!inputFile.is_open()) {
        cerr << "Error opening the file!" << endl;
        return 1;
    }

    auto start_time = std::chrono::high_resolution_clock::now();
    int V, E;
    inputFile >> V >> E;

    vector<vector<int>> adj(V);
    for (int i = 0; i < E; ++i) {
        int u, v_node;
        inputFile >> u >> v_node;
        adj[u].push_back(v_node);
    }

    inputFile.close();

    printSCCs(V, adj);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "\nTotal Execution Time: " << duration.count() << " ms" << std::endl;
    double time = std::chrono::duration<double>(end_time - start_time).count(); // time in seconds
    double bytes_accessed = 4.0 * (2.0 * E + 2.0 * V); // Approximation in bytes
    double gb_accessed = bytes_accessed / (1024.0 * 1024.0 * 1024.0); // Convert to GB
    double throughput_gbps = gb_accessed / time;

    cout << "Estimated Memory Access: " << gb_accessed << " GB" << endl;
    cout << "Estimated Memory Throughput: " << throughput_gbps << " GB/s" << endl;

    return 0;
}