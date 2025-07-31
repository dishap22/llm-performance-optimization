#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <omp.h>
#include <chrono>

class Graph {
public:
    Graph(const std::string& filename) {
        load_from_file(filename);
    }

    void findSCCs() {
        std::vector<bool> removed(node_count, false);
        std::vector<std::vector<int>> sccs;

        trim_trivial_and_size2(removed, sccs);

        while (true) {
            int pivot = -1;
            for (int i = 0; i < node_count; ++i) {
                if (!removed[i]) {
                    pivot = i;
                    break;
                }
            }
            if (pivot == -1) break;

            std::vector<bool> forward(node_count, false);
            std::vector<bool> backward(node_count, false);

            bfs(pivot, adj, removed, forward);
            bfs(pivot, rev_adj, removed, backward);

            std::vector<int> scc;
            #pragma omp parallel
            {
                std::vector<int> local_scc;
                #pragma omp for nowait
                for (int i = 0; i < node_count; ++i) {
                    if (!removed[i] && forward[i] && backward[i]) {
                        local_scc.push_back(i);
                    }
                }
                #pragma omp critical
                scc.insert(scc.end(), local_scc.begin(), local_scc.end());
            }

            for (int u : scc) removed[u] = true;
            if (!scc.empty()) sccs.push_back(scc);

            trim_trivial_and_size2(removed, sccs);
        }

        int count = 0;
        for (const auto& comp : sccs) {
            std::cout << "SCC " << ++count << ": ";
            for (int node : comp) std::cout << reverse_node_map[node] << " ";
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
    int node_count = 0;
    std::unordered_map<int, int> node_map;
    std::unordered_map<int, int> reverse_node_map;
    std::vector<std::vector<int>> adj, rev_adj;

    void bfs(int start, const std::vector<std::vector<int>>& graph,
             const std::vector<bool>& removed, std::vector<bool>& visited) {
        visited[start] = true;
        std::queue<int> q;
        q.push(start);

        while (!q.empty()) {
            int frontier_size = q.size();
            std::vector<int> next_frontier;

            #pragma omp parallel
            {
                std::vector<int> local_frontier;
                #pragma omp for nowait
                for (int i = 0; i < frontier_size; ++i) {
                    int u;
                    #pragma omp critical
                    {
                        u = q.front();
                        q.pop();
                    }

                    for (int v : graph[u]) {
                        if (!removed[v] && !visited[v]) {
                            if (!visited[v]) {
                                visited[v] = true;
                                local_frontier.push_back(v);
                            }
                        }
                    }
                }

                #pragma omp critical
                for (int v : local_frontier) q.push(v);
            }
        }
    }

    void trim_trivial_and_size2(std::vector<bool>& removed, std::vector<std::vector<int>>& sccs) {
        bool changed = true;
        while (changed) {
            changed = false;

            std::vector<int> in_deg(node_count, 0), out_deg(node_count, 0);
            for (int u = 0; u < node_count; ++u) {
                if (removed[u]) continue;
                for (int v : adj[u]) {
                    if (!removed[v]) {
                        out_deg[u]++;
                        in_deg[v]++;
                    }
                }
            }

            #pragma omp parallel for
            for (int i = 0; i < node_count; ++i) {
                if (!removed[i] && (in_deg[i] == 0 || out_deg[i] == 0)) {
                    removed[i] = true;
                    #pragma omp critical
                    sccs.push_back({i});
                    changed = true;
                }
            }
        }

        // Size-2 SCCs
        #pragma omp parallel for schedule(dynamic)
        for (int u = 0; u < node_count; ++u) {
            if (removed[u] || adj[u].size() != 1) continue;
            int v = adj[u][0];
            if (!removed[v] && adj[v].size() == 1 && adj[v][0] == u) {
                removed[u] = true;
                removed[v] = true;
                #pragma omp critical
                sccs.push_back({u, v});
            }
        }
    }

    void load_from_file(const std::string& filename) {
        std::ifstream infile(filename);
        if (!infile.is_open()) {
            std::cerr << "Error: Cannot open file " << filename << "\n";
            exit(1);
        }

        std::string line;
        std::vector<std::pair<int, int>> edges;
        while (std::getline(infile, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::stringstream ss(line);
            int u, v;
            if (ss >> u >> v) {
                if (node_map.count(u) == 0) {
                    node_map[u] = node_count;
                    reverse_node_map[node_count++] = u;
                }
                if (node_map.count(v) == 0) {
                    node_map[v] = node_count;
                    reverse_node_map[node_count++] = v;
                }
                edges.emplace_back(node_map[u], node_map[v]);
            }
        }

        adj.resize(node_count);
        rev_adj.resize(node_count);
        for (auto [u, v] : edges) {
            adj[u].push_back(v);
            rev_adj[v].push_back(u);
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

