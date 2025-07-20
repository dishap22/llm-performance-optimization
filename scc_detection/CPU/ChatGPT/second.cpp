#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <mutex>
#include <omp.h>
#include <chrono>

class Graph {
public:
    Graph(const std::string& filename) {
        load_from_file(filename);
    }

    void findSCCs() {
        int n = node_count;
        std::vector<bool> visited(n, false);
        std::stack<int> finish_stack;
        std::mutex stack_mutex;

        // Phase 1: First DFS for finishing times
        #pragma omp parallel for shared(visited)
        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                std::stack<int> local_stack;
                dfs1(i, visited, local_stack);
                std::lock_guard<std::mutex> lock(stack_mutex);
                while (!local_stack.empty()) {
                    finish_stack.push(local_stack.top());
                    local_stack.pop();
                }
            }
        }

        // Phase 2: Transpose the graph
        std::vector<std::vector<int>> transposed(n);
        #pragma omp parallel for
        for (int u = 0; u < n; ++u) {
            for (int v : adj[u]) {
                #pragma omp critical
                transposed[v].push_back(u);
            }
        }

        // Phase 3: Second DFS on transposed graph
        std::vector<bool> visited2(n, false);
        std::vector<std::vector<int>> sccs;
        std::mutex scc_mutex;

        #pragma omp parallel
        {
            std::vector<int> local_component;
            std::stack<int> local_stack;

            while (true) {
                int node = -1;
                #pragma omp critical
                {
                    while (!finish_stack.empty()) {
                        int top = finish_stack.top();
                        finish_stack.pop();
                        if (!visited2[top]) {
                            node = top;
                            break;
                        }
                    }
                }

                if (node == -1) break;

                dfs2(node, visited2, transposed, local_component, local_stack);

                #pragma omp critical
                sccs.push_back(local_component);
                local_component.clear();
            }
        }

        // Output
        int scc_count = 0;
        for (const auto& comp : sccs) {
            std::cout << "SCC " << ++scc_count << ": ";
            for (int node : comp)
                std::cout << reverse_node_map[node] << " ";
            std::cout << "\n";
        }
        std::cout << "\nTotal Strongly Connected Components: " << scc_count << "\n";
    }
    int getNodeCount() const { return node_count; }
    int getEdgeCount() const {
        int total = 0;
        for (const auto& neighbors : adj)
            total += neighbors.size();
        return total;
    }

private:
    std::unordered_map<int, int> node_map;
    std::unordered_map<int, int> reverse_node_map;
    int node_count = 0;
    std::vector<std::vector<int>> adj;

    void load_from_file(const std::string& filename) {
        std::ifstream infile(filename);
        if (!infile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << "\n";
            exit(1);
        }

        std::string line;
        std::vector<std::pair<int, int>> edges;
        while (std::getline(infile, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            int u, v;
            if (ss >> u >> v) {
                edges.emplace_back(u, v);
                if (node_map.find(u) == node_map.end()) {
                    node_map[u] = node_count;
                    reverse_node_map[node_count++] = u;
                }
                if (node_map.find(v) == node_map.end()) {
                    node_map[v] = node_count;
                    reverse_node_map[node_count++] = v;
                }
            }
        }

        adj.resize(node_count);
        for (auto& [u, v] : edges) {
            adj[node_map[u]].push_back(node_map[v]);
        }
    }

    void dfs1(int u, std::vector<bool>& visited, std::stack<int>& finish_stack) {
        std::stack<int> s;
        s.push(u);
        while (!s.empty()) {
            int node = s.top();
            if (!visited[node]) {
                visited[node] = true;
                for (int v : adj[node]) {
                    if (!visited[v])
                        s.push(v);
                }
            } else {
                s.pop();
                finish_stack.push(node);
            }
        }
    }

    void dfs2(int u, std::vector<bool>& visited, std::vector<std::vector<int>>& transposed,
              std::vector<int>& component, std::stack<int>& s) {
        s.push(u);
        while (!s.empty()) {
            int node = s.top();
            s.pop();
            if (!visited[node]) {
                visited[node] = true;
                component.push_back(node);
                for (int v : transposed[node]) {
                    if (!visited[v])
                        s.push(v);
                }
            }
        }
    }
};

int main() {
    std::string filename;
    std::cout << "Enter input file path: ";
    std::cin >> filename;

    // Start timer
    auto start_time = std::chrono::high_resolution_clock::now();

    Graph g(filename);
    g.findSCCs();

    // Stop timer
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\nTotal Execution Time: " << duration.count() << " ms" << std::endl;
    double time_seconds = duration.count() / 1000.0;
    size_t bytes_accessed = 4L * (2 * g.getEdgeCount() + 6 * g.getNodeCount()); // Approximation
    double gb_accessed = static_cast<double>(bytes_accessed) / (1024.0 * 1024.0 * 1024.0);
    double gbps = gb_accessed / time_seconds;

    std::cout << "Estimated Memory Accessed: " << gb_accessed << " GB" << std::endl;
    std::cout << "Estimated Memory Throughput: " << gbps << " GB/s" << std::endl;

    return 0;
}

