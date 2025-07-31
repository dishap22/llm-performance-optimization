#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
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

        // First DFS on original graph to get finishing order
        for (int i = 0; i < n; ++i) {
            if (!visited[i]) {
                dfs1(i, visited, finish_stack);
            }
        }

        // Transpose the graph
        std::vector<std::vector<int>> transposed(n);
        for (int u = 0; u < n; ++u) {
            for (int v : adj[u]) {
                transposed[v].push_back(u);
            }
        }

        // Second DFS in finishing order on transposed graph
        std::fill(visited.begin(), visited.end(), false);
        int scc_count = 0;
        while (!finish_stack.empty()) {
            int node = finish_stack.top();
            finish_stack.pop();

            if (!visited[node]) {
                std::vector<int> component;
                dfs2(node, visited, transposed, component);
                ++scc_count;
                std::cout << "SCC " << scc_count << ": ";
                for (int x : component)
                    std::cout << reverse_node_map[x] << " ";
                std::cout << "\n";
            }
        }

        std::cout << "\nTotal Strongly Connected Components: " << scc_count << std::endl;
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
                    reverse_node_map[node_count] = u;
                    ++node_count;
                }
                if (node_map.find(v) == node_map.end()) {
                    node_map[v] = node_count;
                    reverse_node_map[node_count] = v;
                    ++node_count;
                }
            }
        }

        adj.resize(node_count);
        for (auto& [u, v] : edges) {
            adj[node_map[u]].push_back(node_map[v]);
        }
    }

    void dfs1(int u, std::vector<bool>& visited, std::stack<int>& finish_stack) {
        visited[u] = true;
        for (int v : adj[u]) {
            if (!visited[v]) dfs1(v, visited, finish_stack);
        }
        finish_stack.push(u);
    }

    void dfs2(int u, std::vector<bool>& visited, std::vector<std::vector<int>>& transposed, std::vector<int>& component) {
        visited[u] = true;
        component.push_back(u);
        for (int v : transposed[u]) {
            if (!visited[v]) dfs2(v, visited, transposed, component);
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
