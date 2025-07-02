// Final SCC Method2 Implementation with Trim, Trim2, and Parallel FW-BW Recursion
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

struct Graph {
    int num_nodes;
    int num_edges;
    vector<int> forward_offsets;
    vector<int> forward_edges;
    vector<int> backward_offsets;
    vector<int> backward_edges;
};

Graph load_graph_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    std::vector<std::pair<int, int>> raw_edge_list;
    std::unordered_map<int, int> node_map;
    int next_node_id = 0;
    int u_orig, v_orig;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        if (ss >> u_orig >> v_orig) {
            raw_edge_list.push_back({u_orig, v_orig});
            if (node_map.find(u_orig) == node_map.end())
                node_map[u_orig] = next_node_id++;
            if (node_map.find(v_orig) == node_map.end())
                node_map[v_orig] = next_node_id++;
        }
    }
    file.close();

    Graph g;
    g.num_nodes = node_map.size();
    g.num_edges = raw_edge_list.size();

    std::vector<std::pair<int, int>> remapped_edge_list;
    remapped_edge_list.reserve(g.num_edges);
    for (const auto& edge : raw_edge_list) {
        remapped_edge_list.push_back({node_map[edge.first], node_map[edge.second]});
    }

    g.forward_offsets.assign(g.num_nodes + 1, 0);
    g.forward_edges.resize(g.num_edges);
    std::vector<int> fwd_degrees(g.num_nodes, 0);
    for (const auto& edge : remapped_edge_list)
        fwd_degrees[edge.first]++;
    for (int i = 1; i <= g.num_nodes; ++i)
        g.forward_offsets[i] = g.forward_offsets[i - 1] + fwd_degrees[i - 1];
    auto fwd_counters = g.forward_offsets;
    for (const auto& edge : remapped_edge_list)
        g.forward_edges[fwd_counters[edge.first]++] = edge.second;

    g.backward_offsets.assign(g.num_nodes + 1, 0);
    g.backward_edges.resize(g.num_edges);
    std::vector<int> bwd_degrees(g.num_nodes, 0);
    for (const auto& edge : remapped_edge_list)
        bwd_degrees[edge.second]++;
    for (int i = 1; i <= g.num_nodes; ++i)
        g.backward_offsets[i] = g.backward_offsets[i - 1] + bwd_degrees[i - 1];
    auto bwd_counters = g.backward_offsets;
    for (const auto& edge : remapped_edge_list)
        g.backward_edges[bwd_counters[edge.second]++] = edge.first;

    std::cout << "Graph loaded: " << g.num_nodes << " unique nodes, " << g.num_edges << " edges." << std::endl;
    return g;
}

void parallel_bfs_csr(const Graph& g, const vector<bool>& active, int start, vector<bool>& visited, vector<int>& reachable, bool forward) {
    queue<int> q;
    visited[start] = true;
    q.push(start);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        reachable.push_back(u);
        const auto& offsets = forward ? g.forward_offsets : g.backward_offsets;
        const auto& edges = forward ? g.forward_edges : g.backward_edges;
        for (int i = offsets[u]; i < offsets[u + 1]; ++i) {
            int v = edges[i];
            if (active[v] && !visited[v]) {
                visited[v] = true;
                q.push(v);
            }
        }
    }
}

void trim(const Graph& g, vector<bool>& active, vector<vector<int>>& sccs) {
    bool changed = true;
    while (changed) {
        changed = false;
        #pragma omp parallel for schedule(dynamic)
        for (int u = 0; u < g.num_nodes; ++u) {
            if (!active[u]) continue;
            bool has_out = false, has_in = false;
            for (int i = g.forward_offsets[u]; i < g.forward_offsets[u + 1]; ++i)
                if (active[g.forward_edges[i]]) has_out = true;
            for (int i = g.backward_offsets[u]; i < g.backward_offsets[u + 1]; ++i)
                if (active[g.backward_edges[i]]) has_in = true;
            if (!has_in || !has_out) {
                active[u] = false;
                #pragma omp critical
                sccs.push_back({u});
                changed = true;
            }
        }
    }
}

void trim2(const Graph& g, vector<bool>& active, vector<vector<int>>& sccs) {
    #pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < g.num_nodes; ++u) {
        if (!active[u]) continue;
        int out_count = 0, only_out = -1;
        for (int i = g.forward_offsets[u]; i < g.forward_offsets[u + 1]; ++i) {
            int v = g.forward_edges[i];
            if (active[v]) {
                only_out = v;
                out_count++;
                if (out_count > 1) break;
            }
        }
        if (out_count != 1 || !active[only_out]) continue;

        int in_count = 0;
        for (int i = g.backward_offsets[u]; i < g.backward_offsets[u + 1]; ++i)
            if (active[g.backward_edges[i]]) in_count++;

        if (in_count == 1) {
            #pragma omp critical
            sccs.push_back({u, only_out});
            active[u] = false;
            active[only_out] = false;
        }
    }
}

void fw_bw(const Graph& g, vector<bool>& active, int pivot, vector<int>& scc) {
    vector<bool> fw_visited(g.num_nodes, false), bw_visited(g.num_nodes, false);
    vector<int> fw, bw;
    parallel_bfs_csr(g, active, pivot, fw_visited, fw, true);
    parallel_bfs_csr(g, active, pivot, bw_visited, bw, false);

    unordered_set<int> fw_set(fw.begin(), fw.end());
    for (int v : bw) {
        if (fw_set.count(v)) {
            scc.push_back(v);
            active[v] = false;
        }
    }
}

void method2(const Graph& g) {
    vector<bool> active(g.num_nodes, true);
    vector<vector<int>> sccs;

    trim(g, active, sccs);
    trim2(g, active, sccs);

    #pragma omp parallel
    {
        vector<vector<int>> local_sccs;
        #pragma omp for schedule(dynamic)
        for (int i = 0; i < g.num_nodes; ++i) {
            if (!active[i]) continue;
            vector<int> scc;
            fw_bw(g, active, i, scc);
            if (!scc.empty()) {
                #pragma omp critical
                sccs.push_back(scc);
            }
        }
    }

    cout << "SCCs found: " << sccs.size() << endl;
    // for (auto& scc : sccs) {
    //     for (int v : scc) cout << v << " ";
    //     cout << endl;
    // }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <snap_file.txt>" << std::endl;
        return 1;
    }
    Graph g = load_graph_from_file(argv[1]);
    method2(g);
    return 0;
}
