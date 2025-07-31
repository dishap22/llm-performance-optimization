#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <algorithm>
#include <omp.h> // Include the OpenMP header
#include <chrono> // For measuring execution time

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
    stack<int> s;
    vector<bool> visited(V, false);

    // Step 1: Fill vertices in stack according to their finishing times.
    // This step remains sequential as the finishing order is crucial and depends on a specific traversal order.
    for (int i = 0; i < V; ++i) {
        if (!visited[i]) {
            fillOrder(i, visited, adj, s);
        }
    }

    // Step 2: Create a transposed graph.
    // This loop is parallelized. Each iteration is independent.
    vector<vector<int>> adjTranspose(V);
    #pragma omp parallel for
    for (int v = 0; v < V; ++v) {
        for (int i : adj[v]) {
            // A critical section is needed because multiple threads might try to 
            // push to the same adjTranspose[i] vector simultaneously, which is not thread-safe.
            #pragma omp critical
            adjTranspose[i].push_back(v);
        }
    }

    // Step 3: Process all vertices in order defined by the stack to find SCCs.
    // This part is parallelized by processing nodes from the stack concurrently.
    fill(visited.begin(), visited.end(), false);
    
    vector<int> processing_order;
    processing_order.reserve(V);
    while(!s.empty()){
        processing_order.push_back(s.top());
        s.pop();
    }
    
    vector<vector<int>> scc_list;

    #pragma omp parallel
    {
        vector<vector<int>> private_scc_list; // Each thread accumulates its own found components

        // Use a dynamic schedule as SCCs can have vastly different sizes, leading to imbalanced work.
        #pragma omp for schedule(dynamic)
        for (size_t i = 0; i < processing_order.size(); ++i) {
            int v = processing_order[i];
            
            // Check if the vertex has been visited. If not, it's the root of a new SCC.
            // This check needs to be inside a critical section to prevent a race condition where
            // multiple threads might find the same unvisited component root and start a DFS for it.
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

        // Merge the private lists from each thread into the main list inside a critical section.
        #pragma omp critical(scc_merge)
        {
            scc_list.insert(scc_list.end(), private_scc_list.begin(), private_scc_list.end());
        }
    }
    
    // Print the collected SCCs sequentially
    cout << "Strongly Connected Components in the given graph:" << endl;
    int count = 0;
    for (const auto& component : scc_list) {
        cout << "Strongly Connected Component " << ++count << ": ";
        for (int node : component) {
            cout << node << " ";
        }
        cout << endl;
    }
    cout << "\nTotal Strongly Connected Components: " << scc_list.size() << endl;
}

int main() {
    ifstream inputFile("web-NotreDame.txt");
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
    double time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time).count();
    double bytes_accessed = 4.0 * (2.0 * E + 2.0 * V); // Approximation in bytes
    double gb_accessed = bytes_accessed / (1024.0 * 1024.0 * 1024.0); // Convert to GB
    double throughput_gbps = gb_accessed / time;

    cout << "Estimated Memory Access: " << gb_accessed << " GB" << endl;
    cout << "Estimated Memory Throughput: " << throughput_gbps << " GB/s" << endl;


    return 0;
}