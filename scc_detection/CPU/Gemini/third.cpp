#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <algorithm>
#include <omp.h> // Include the OpenMP header
#include <chrono>

// Standard namespace for cleaner code
using namespace std;

// First DFS to fill the stack with vertices in order of finishing times (sequential)
void fillOrder(int v, vector<bool>& visited, const vector<vector<int>>& adj, stack<int>& s) {
    visited[v] = true;
    for (int i : adj[v]) {
        if (!visited[i]) {
            fillOrder(i, visited, adj, s);
        }
    }
    s.push(v);
}

// Second DFS to find and print SCCs (run in parallel for different components)
void dfs(int v, vector<bool>& visited, const vector<vector<int>>& adjTranspose, vector<int>& component) {
    visited[v] = true;
    component.push_back(v);
    for (int i : adjTranspose[v]) {
        if (!visited[i]) {
            dfs(i, visited, adjTranspose, component);
        }
    }
}

// Function to find and print all strongly connected components
void printSCCs(int V, const vector<vector<int>>& adj) {
    vector<vector<int>> scc_list;
    vector<bool> is_processed(V, false);
    int remaining_nodes = V;

    // --- ENHANCEMENT 1: Trimming Trivial SCCs ---
    // Pre-process the graph to find and remove nodes with in-degree or out-degree of 0.
    // These nodes are guaranteed to be trivial SCCs of size 1.
    if (V > 0) {
        vector<int> in_degree(V, 0);
        for (int u = 0; u < V; ++u) {
            for (int v : adj[u]) {
                in_degree[v]++;
            }
        }

        for (int i = 0; i < V; ++i) {
            // A node with no incoming or no outgoing edges cannot be part of a multi-node cycle.
            if (in_degree[i] == 0 || adj[i].empty()) {
                scc_list.push_back({i});
                is_processed[i] = true;
                remaining_nodes--;
            }
        }
    }

    // Proceed with the full algorithm only if there are non-trivial nodes left.
    if (remaining_nodes > 0) {
        stack<int> s;
        vector<bool> visited = is_processed; // Start with trimmed nodes already "visited"

        // Step 1: Fill stack only for the remaining (non-trivial) subgraph.
        for (int i = 0; i < V; ++i) {
            if (!visited[i]) {
                fillOrder(i, visited, adj, s);
            }
        }

        // Step 2: Create a transposed graph.
        vector<vector<int>> adjTranspose(V);
        #pragma omp parallel for
        for (int v = 0; v < V; ++v) {
            if (is_processed[v]) continue; // Optimization: skip processed nodes
            for (int i : adj[v]) {
                #pragma omp critical
                adjTranspose[i].push_back(v);
            }
        }

        // Step 3: Find SCCs in the remaining subgraph.
        fill(visited.begin(), visited.end(), false);
        for(int i = 0; i < V; ++i) {
            if(is_processed[i]) visited[i] = true;
        }

        vector<int> processing_order;
        // --- ENHANCEMENT 2: Memory Allocation Optimization ---
        processing_order.reserve(s.size()); // Pre-allocate memory
        while (!s.empty()) {
            processing_order.push_back(s.top());
            s.pop();
        }

        #pragma omp parallel
        {
            vector<vector<int>> private_scc_list;

            #pragma omp for schedule(dynamic)
            for (size_t i = 0; i < processing_order.size(); ++i) {
                int v = processing_order[i];
                if (!visited[v]) {
                    bool process_this_node = false;
                    #pragma omp critical(scc_check)
                    {
                        if (!visited[v]) {
                            process_this_node = true;
                        }
                    }
                    if (process_this_node) {
                        vector<int> component;
                        dfs(v, visited, adjTranspose, component);
                        private_scc_list.push_back(component);
                    }
                }
            }

            #pragma omp critical(scc_merge)
            {
                scc_list.insert(scc_list.end(), private_scc_list.begin(), private_scc_list.end());
            }
        }
    }

    // Print all collected SCCs (both trivial and complex).
    cout << "Strongly Connected Components in the given graph:" << endl;
    int count = 0;
    for (const auto& component : scc_list) {
        cout << "Strongly Connected Component " << ++count << ": ";
        // Sort for consistent output
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
    double time = std::chrono::duration<double>(end_time - start_time).count();
    double bytes_accessed = 4.0 * (2.0 * E + 2.0 * V); // Approximation in bytes
    double gb_accessed = bytes_accessed / (1024.0 * 1024.0 * 1024.0); // Convert to GB
    double throughput_gbps = gb_accessed / time;

    cout << "Estimated Memory Access: " << gb_accessed << " GB" << endl;
    cout << "Estimated Memory Throughput: " << throughput_gbps << " GB/s" << endl;


    return 0;
}