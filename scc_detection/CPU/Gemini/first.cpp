#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <algorithm>
#include <chrono> // For measuring execution time

// Standard namespace for cleaner code
using namespace std;

// First DFS to fill the stack with vertices in order of finishing times
void fillOrder(int v, vector<bool>& visited, const vector<vector<int>>& adj, stack<int>& s) {
    visited[v] = true;
    for (int i : adj[v]) {
        if (!visited[i]) {
            fillOrder(i, visited, adj, s);
        }
    }
    s.push(v);
}

// Second DFS to find and print SCCs
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

    // Step 1: Fill vertices in stack according to their finishing times
    for (int i = 0; i < V; ++i) {
        if (!visited[i]) {
            fillOrder(i, visited, adj, s);
        }
    }

    // Step 2: Create a transposed graph
    vector<vector<int>> adjTranspose(V);
    for (int v = 0; v < V; ++v) {
        for (int i : adj[v]) {
            adjTranspose[i].push_back(v);
        }
    }

    // Step 3: Process all vertices in order defined by the stack
    fill(visited.begin(), visited.end(), false);
    int count = 0;
    while (!s.empty()) {
        int v = s.top();
        s.pop();

        if (!visited[v]) {
            vector<int> component;
            dfs(v, visited, adjTranspose, component);
            cout << "Strongly Connected Component " << ++count << ": ";
            for (int node : component) {
                cout << node << " ";
            }
            cout << endl;
        }
    }
    cout << "\nTotal Strongly Connected Components: " << count << endl;
}

int main() {
    ifstream inputFile("Wiki-Vote.txt");
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

    cout << "Strongly Connected Components in the given graph:" << endl;
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