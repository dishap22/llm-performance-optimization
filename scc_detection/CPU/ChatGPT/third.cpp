#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <omp.h>
#include <chrono>
#include <atomic>
std::atomic<long long> edge_traversals(0);

class Graph {
public:
    Graph(const std::string& filename) {
        load_from_file(filename);
    }

    void findSCCs() {
        // Phase 1: Remove trivial SCCs (no in or no out)
        std::vector<bool> removed(node_count, false);
        std::vector<std::vector<int>> trivial_sccs;

        std::vector<int> indegree(node_count, 0), outdegree(node_count, 0);
        for (int u = 0; u < node_count; ++u) {
            for (int v : adj[u]) {
                outdegree[u]++;
                indegree[v]++;
            }
        }

        std::queue<int> q;
        for (int i = 0; i < node_count; ++i) {
            if (indegree[i] == 0 || outdegree[i] == 0) {
                q.push(i);
                removed[i] = true;
                trivial_sccs.push_back({i});
            }
        }

        // Remove chains of trivial SCCs
        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : adj[u]) {
                if (!removed[v]) {
                    indegree[v]--;
                    if (indegree[v] == 0) {
                        q.push(v);
                        removed[v] = true;
                        trivial_sccs.push_back({v});
                    }
                }
            }
            for (int v : rev_adj[u]) {
                if (!removed[v]) {
                    outdegree[v]--;
                    if (outdegree[v] == 0) {
                        q.push(v);
                        removed[v] = true;
                        trivial_sccs.push_back({v});
                    }
                }
            }
        }

        // Phase 2: Kosaraju’s on remaining nodes
        std::vector<bool> visited(node_count, false);
        std::stack<int> finish_stack;

        for (int i = 0; i < node_count; ++i) {
            edge_traversals++;
            if (!removed[i] && !visited[i]) {
                dfs1(i, visited, finish_stack);
            }
        }

        std::vector<std::vector<int>> sccs;
        std::fill(visited.begin(), visited.end(), false);

        while (!finish_stack.empty()) {
            int node = finish_stack.top(); finish_stack.pop();
            edge_traversals++;  
            if (!removed[node] && !visited[node]) {
                std::vector<int> component;
                dfs2(node, visited, component);
                sccs.push_back(component);
            }
        }

        // Output all SCCs
        int count = 0;
        for (auto& comp : trivial_sccs) {
            std::cout << "SCC " << ++count << ": ";
            for (int node : comp)
                std::cout << reverse_node_map[node] << " ";
            std::cout << "\n";
        }

        for (auto& comp : sccs) {
            std::cout << "SCC " << ++count << ": ";
            for (int node : comp)
                std::cout << reverse_node_map[node] << " ";
            std::cout << "\n";
        }

        std::cout << "\nTotal Strongly Connected Components: " << count << "\n";
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

    std::vector<std::vector<int>> adj;       // forward
    std::vector<std::vector<int>> rev_adj;   // backward

    void dfs1(int u, std::vector<bool>& visited, std::stack<int>& finish_stack) {
        visited[u] = true;
        for (int v : adj[u]) {
            edge_traversals++;
            if (!visited[v]) dfs1(v, visited, finish_stack);
        }
        finish_stack.push(u);
    }

    void dfs2(int u, std::vector<bool>& visited, std::vector<int>& component) {
        visited[u] = true;
        component.push_back(u);
        for (int v : rev_adj[u]) {
            edge_traversals++;
            if (!visited[v]) dfs2(v, visited, component);
        }
    }

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
        rev_adj.resize(node_count);
        for (auto& [u, v] : edges) {
            int uid = node_map[u];
            int vid = node_map[v];
            adj[uid].push_back(vid);
            rev_adj[vid].push_back(uid);
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
    double teps = edge_traversals / duration.count();
    // std::cout << "\nTotal Execution Time: " << duration.count() << " ms" << std::endl;
    double time_seconds = duration.count() / 1000.0;
    size_t bytes_accessed = 4L * (2 * g.getEdgeCount() + 6 * g.getNodeCount()); // Approximation
    double gb_accessed = static_cast<double>(bytes_accessed) / (1024.0 * 1024.0 * 1024.0);
    double gbps = gb_accessed / time_seconds;

    std::cout << "Estimated Memory Accessed: " << gb_accessed << " GB" << std::endl;
    std::cout << "Estimated Memory Throughput: " << gbps << " GB/s" << std::endl;

    return 0;
}

