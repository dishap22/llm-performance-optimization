#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <atomic>
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
        std::vector<bool> removed(node_count, false);
        std::vector<std::vector<int>> all_sccs;

        trim_trivial_and_size2(removed, all_sccs);

        std::vector<int> labels(node_count, -1);
        compute_wccs_hong_style(removed, labels);

        int max_label = *max_element(labels.begin(), labels.end());

        #pragma omp parallel for schedule(dynamic)
        for (int w = 0; w <= max_label; ++w) {
            std::vector<int> wcc;
            for (int i = 0; i < node_count; ++i) {
                if (labels[i] == w) wcc.push_back(i);
            }
            if (wcc.empty()) continue;

            std::vector<std::vector<int>> sccs;
            find_sccs_parallel(wcc, sccs);

            #pragma omp critical
            all_sccs.insert(all_sccs.end(), sccs.begin(), sccs.end());
        }

        int count = 0;
        for (const auto& comp : all_sccs) {
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

    void compute_wccs_hong_style(const std::vector<bool>& removed, std::vector<int>& label) {
        #pragma omp parallel for
        for (int i = 0; i < node_count; ++i) {
            if (!removed[i]) label[i] = i;
        }

        bool changed = true;
        while (changed) {
            changed = false;
            #pragma omp parallel for
            for (int u = 0; u < node_count; ++u) {
                if (label[u] != -1) {
                    for (int v : adj[u]) {
                        if (label[v] != -1 && label[v] > label[u]) {
                            #pragma omp critical
                            {
                                if (label[v] > label[u]) {
                                    label[v] = label[u];
                                    changed = true;
                                }
                            }
                        }
                    }
                    for (int v : rev_adj[u]) {
                        if (label[v] != -1 && label[v] > label[u]) {
                            #pragma omp critical
                            {
                                if (label[v] > label[u]) {
                                    label[v] = label[u];
                                    changed = true;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Normalize labels
        std::unordered_map<int, int> remap;
        int id = 0;
        for (int& v : label) {
            if (v != -1 && remap.find(v) == remap.end()) {
                remap[v] = id++;
            }
            if (v != -1) v = remap[v];
        }
    }

    void bfs_atomic(int start, const std::vector<std::vector<int>>& graph,
                    const std::vector<bool>& in_set, std::vector<std::atomic<bool>>& visited) {
        std::queue<int> q;
        q.push(start);
        visited[start] = true;

        while (!q.empty()) {
            int u = q.front(); q.pop();
            for (int v : graph[u]) {
                edge_traversals++;
                if (in_set[v] && !visited[v].exchange(true)) {
                    q.push(v);
                }
            }
        }
    }

    void find_sccs_parallel(const std::vector<int>& nodes, std::vector<std::vector<int>>& sccs) {
        std::vector<bool> claimed(node_count, false);
        std::vector<bool> in_set(node_count, false);
        for (int v : nodes) in_set[v] = true;

        while (true) {
            int pivot = -1;
            for (int v : nodes) {
                if (in_set[v] && !claimed[v]) {
                    pivot = v;
                    break;
                }
            }
            if (pivot == -1) break;

            std::vector<std::atomic<bool>> fwd(node_count);
            std::vector<std::atomic<bool>> bwd(node_count);
            for (int i = 0; i < node_count; ++i) fwd[i] = bwd[i] = false;

            bfs_atomic(pivot, adj, in_set, fwd);
            bfs_atomic(pivot, rev_adj, in_set, bwd);

            std::vector<int> scc;
            for (int v : nodes) {
                if (in_set[v] && fwd[v] && bwd[v]) {
                    scc.push_back(v);
                    claimed[v] = true;
                    in_set[v] = false;
                }
            }
            if (!scc.empty()) sccs.push_back(scc);
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
    double teps = edge_traversals / duration.count();
    std::cout << "\nTotal Execution Time: " << duration.count() << " ms" << std::endl;
    double time_seconds = duration.count() / 1000.0;
    size_t bytes_accessed = 4L * (2 * g.getEdgeCount() + 6 * g.getNodeCount()); // Approximation
    double gb_accessed = static_cast<double>(bytes_accessed) / (1024.0 * 1024.0 * 1024.0);
    double gbps = gb_accessed / time_seconds;

    std::cout << "Estimated Memory Accessed: " << gb_accessed << " GB" << std::endl;
    std::cout << "Estimated Memory Throughput: " << gbps << " GB/s" << std::endl;

    return 0;
}

