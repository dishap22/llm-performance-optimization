#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>
#include <set>
#include <unordered_map>
#include <omp.h>

// Graph struct remains the same
struct Graph {
    int num_nodes;
    int num_edges;
    std::vector<int> forward_offsets;
    std::vector<int> forward_edges;
    std::vector<int> backward_offsets;
    std::vector<int> backward_edges;
};

// ... (Paste the entire SccSolver class from the previous answer here) ...
class SccSolver {
public:
    SccSolver(Graph& g) : graph(g) {
        colors.resize(g.num_nodes, 0);
        marks.resize(g.num_nodes, false);
        sccs.reserve(g.num_nodes);
        next_color = 1;
    }

    void run_method2() {
        std::cout << "Phase 1: Parallel Trims, FWBW, and WCC detection" << std::endl;
        par_trim("Initial Par-Trim");
        par_fwbw();
        par_trim("Par-Trim post-FWBW");
        par_trim2();
        par_trim("Par-Trim post-Trim2");
        par_wcc();
        std::cout << "Phase 2: Parallelism in Recursion" << std::endl;
        #pragma omp parallel
        {
            process_work_queue();
        }
        std::cout << "SCC detection complete." << std::endl;
    }

    const std::vector<std::vector<int>>& get_sccs() const {
        return sccs;
    }

private:
    Graph& graph;
    std::vector<int> colors;
    std::vector<bool> marks;
    std::vector<std::vector<int>> sccs;
    std::atomic<int> next_color;
    std::deque<int> work_queue;
    std::mutex queue_mutex;

    void par_trim(const std::string& phase_name) {
        bool changed;
        do {
            changed = false;
            std::vector<int> new_sccs;
            #pragma omp parallel for
            for (int u = 0; u < graph.num_nodes; ++u) {
                if (marks[u]) continue;
                int in_degree = 0, out_degree = 0;
                for (int i = graph.backward_offsets[u]; i < graph.backward_offsets[u + 1]; ++i)
                    if (!marks[graph.backward_edges[i]] && colors[graph.backward_edges[i]] == colors[u]) in_degree++;
                for (int i = graph.forward_offsets[u]; i < graph.forward_offsets[u + 1]; ++i)
                    if (!marks[graph.forward_edges[i]] && colors[graph.forward_edges[i]] == colors[u]) out_degree++;
                
                if (in_degree == 0 || out_degree == 0) {
                    if (!marks[u]) {
                       #pragma omp critical
                       {
                           if (!marks[u]) {
                               marks[u] = true;
                               new_sccs.push_back(u);
                               changed = true;
                           }
                       }
                    }
                }
            }
            if(changed) {
                for(int node : new_sccs) sccs.push_back({node});
            }
        } while (changed);
    }

    void parallel_bfs(int start_node, bool forward, std::vector<int>& visited, int target_color) {
        std::vector<int> frontier;
        frontier.push_back(start_node);
        visited[start_node] = 1;

        const auto& offsets = forward ? graph.forward_offsets : graph.backward_offsets;
        const auto& edges = forward ? graph.forward_edges : graph.backward_edges;

        while (!frontier.empty()) {
            std::vector<int> next_frontier;
            #pragma omp parallel for
            for (size_t i = 0; i < frontier.size(); ++i) {
                int u = frontier[i];
                for (int j = offsets[u]; j < offsets[u+1]; ++j) {
                    int v = edges[j];
                    if (!marks[v] && colors[v] == target_color) {
                        if (visited[v] == 0) {
                           #pragma omp atomic write
                           visited[v] = 1;
                           #pragma omp critical
                           next_frontier.push_back(v);
                        }
                    }
                }
            }
            frontier = next_frontier;
        }
    }
    void par_fwbw() {
        int pivot = -1;
        for(int i = 0; i < graph.num_nodes; ++i) if(!marks[i]) { pivot = i; break; }
        if (pivot == -1) return;

        std::vector<int> fw_visited(graph.num_nodes, 0);
        std::vector<int> bw_visited(graph.num_nodes, 0);
        parallel_bfs(pivot, true, fw_visited, colors[pivot]);
        parallel_bfs(pivot, false, bw_visited, colors[pivot]);

        std::vector<int> scc_nodes;
        int c_fw = next_color++;
        int c_bw = next_color++;

        #pragma omp parallel for
        for (int i = 0; i < graph.num_nodes; i++) {
            if (marks[i] || colors[i] != colors[pivot]) continue;
            bool in_fw = fw_visited[i];
            bool in_bw = bw_visited[i];

            if (in_fw && in_bw) {
                #pragma omp critical
                scc_nodes.push_back(i);
                marks[i] = true;
            } else if (in_fw) {
                colors[i] = c_fw;
            } else if (in_bw) {
                colors[i] = c_bw;
            }
        }
        if (!scc_nodes.empty()) sccs.push_back(scc_nodes);
        
        std::lock_guard<std::mutex> lock(queue_mutex);
        work_queue.push_back(colors[pivot]);
        work_queue.push_back(c_fw);
        work_queue.push_back(c_bw);
    }
    
    void par_trim2() {
        std::vector<std::pair<int, int>> new_sccs_pairs;
        #pragma omp parallel for
        for (int u = 0; u < graph.num_nodes; ++u) {
            if (marks[u]) continue;

            if (get_degree(u, true, colors[u]) == 1 && get_degree(u, false, colors[u]) == 1) {
                int v = get_neighbor(u, true, colors[u]);
                if (v != -1 && v == get_neighbor(u, false, colors[u])) {
                    bool found = false;
                    if (get_degree(v, false, colors[v]) == 1) {
                        found = true;
                    } 
                    else if (get_degree(v, true, colors[v]) == 1) {
                        found = true;
                    }
                    if(found){
                       #pragma omp critical
                       {
                           if (!marks[u] && !marks[v]) {
                               marks[u] = marks[v] = true;
                               new_sccs_pairs.push_back({u, v});
                           }
                       }
                    }
                }
            }
        }
        for(const auto& p : new_sccs_pairs) sccs.push_back({p.first, p.second});
    }
    
    void par_wcc() {
        std::vector<std::atomic<int>> wcc_ids(graph.num_nodes);
        #pragma omp parallel for
        for (int i = 0; i < graph.num_nodes; ++i) wcc_ids[i] = i;

        bool changed = true;
        while (changed) {
            changed = false;
            #pragma omp parallel for
            for (int u = 0; u < graph.num_nodes; ++u) {
                if (marks[u]) continue;
                auto propagate = [&](int neighbor) {
                    if (!marks[neighbor] && colors[u] == colors[neighbor]) {
                        int id_u = wcc_ids[u].load();
                        int id_v = wcc_ids[neighbor].load();
                        if (id_u < id_v) {
                            if (wcc_ids[neighbor].compare_exchange_strong(id_v, id_u)) changed = true;
                        } else if (id_v < id_u) {
                            if (wcc_ids[u].compare_exchange_strong(id_u, id_v)) changed = true;
                        }
                    }
                };
                for (int i = graph.forward_offsets[u]; i < graph.forward_offsets[u+1]; ++i) propagate(graph.forward_edges[i]);
                for (int i = graph.backward_offsets[u]; i < graph.backward_offsets[u+1]; ++i) propagate(graph.backward_edges[i]);
            }
        }

        std::lock_guard<std::mutex> lock(queue_mutex);
        work_queue.clear();
        
        #pragma omp parallel
        {
            std::vector<int> local_new_colors;
            #pragma omp for
            for (int i = 0; i < graph.num_nodes; ++i) {
                if (!marks[i] && wcc_ids[i] == i) {
                    int c = next_color++;
                    local_new_colors.push_back(c);
                    for (int j = 0; j < graph.num_nodes; ++j) {
                        if (!marks[j] && wcc_ids[j] == i) colors[j] = c;
                    }
                }
            }
            #pragma omp critical
            for(int c : local_new_colors) work_queue.push_back(c);
        }
    }

    void process_work_queue() {
        while (true) {
            int task_color = -1;
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (work_queue.empty()) break; 
                task_color = work_queue.front();
                work_queue.pop_front();
            }
            if (task_color != -1) recur_fwbw(task_color);
        }
    }

    void recur_fwbw(int color_c) {
        int pivot = -1;
        for (int i = 0; i < graph.num_nodes; ++i) {
            if (!marks[i] && colors[i] == color_c) {
                pivot = i;
                break;
            }
        }
        if (pivot == -1) return;

        std::vector<int> fw_reach, bw_reach;
        std::vector<bool> visited_fw(graph.num_nodes, false);
        dfs_reach(pivot, true, fw_reach, color_c, visited_fw);
        std::vector<bool> visited_bw(graph.num_nodes, false);
        dfs_reach(pivot, false, bw_reach, color_c, visited_bw);

        std::set<int> fw_set(fw_reach.begin(), fw_reach.end());
        std::vector<int> scc_nodes;
        for (int node : bw_reach) {
            if (fw_set.count(node)) {
                scc_nodes.push_back(node);
            }
        }
        
        if (!scc_nodes.empty()) {
            int c_fw = next_color++;
            int c_bw = next_color++;
            
            for (int node : scc_nodes) marks[node] = true;
            
            for (int node : fw_reach) if (!marks[node]) colors[node] = c_fw;
            for (int node : bw_reach) if (!marks[node] && fw_set.find(node) == fw_set.end()) colors[node] = c_bw;

            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                sccs.push_back(scc_nodes);
                work_queue.push_back(c_fw);
                work_queue.push_back(c_bw);
                work_queue.push_back(color_c);
            }
        }
    }
    
    void dfs_reach(int u, bool forward, std::vector<int>& reach, int target_color, std::vector<bool>& visited) {
        visited[u] = true;
        reach.push_back(u);
        const auto& offsets = forward ? graph.forward_offsets : graph.backward_offsets;
        const auto& edges = forward ? graph.forward_edges : graph.backward_edges;
        for (int i = offsets[u]; i < offsets[u+1]; ++i) {
            int v = edges[i];
            if (!marks[v] && colors[v] == target_color && !visited[v]) {
                dfs_reach(v, forward, reach, target_color, visited);
            }
        }
    }
    
    int get_degree(int u, bool forward, int target_color) {
        int degree = 0;
        const auto& offsets = forward ? graph.forward_offsets : graph.backward_offsets;
        const auto& edges = forward ? graph.forward_edges : graph.backward_edges;
        for (int i = offsets[u]; i < offsets[u+1]; ++i) {
            int v = edges[i];
            if (!marks[v] && colors[v] == target_color) degree++;
        }
        return degree;
    }

    int get_neighbor(int u, bool forward, int target_color) {
        const auto& offsets = forward ? graph.forward_offsets : graph.backward_offsets;
        const auto& edges = forward ? graph.forward_edges : graph.backward_edges;
        for (int i = offsets[u]; i < offsets[u+1]; ++i) {
            int v = edges[i];
            if (!marks[v] && colors[v] == target_color) return v;
        }
        return -1;
    }
};

/**
 * @brief Loads a graph from a SNAP-style text file into CSR format.
 * It discovers nodes and remaps them to a 0-indexed contiguous range.
 * @param filename The path to the input graph file.
 * @return A Graph object.
 */
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

    // Read the file line-by-line, skipping comments
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::stringstream ss(line);
        if (ss >> u_orig >> v_orig) {
            raw_edge_list.push_back({u_orig, v_orig});
            if (node_map.find(u_orig) == node_map.end()) {
                node_map[u_orig] = next_node_id++;
            }
            if (node_map.find(v_orig) == node_map.end()) {
                node_map[v_orig] = next_node_id++;
            }
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

    // Build forward graph (CSR)
    g.forward_offsets.assign(g.num_nodes + 1, 0);
    g.forward_edges.resize(g.num_edges);
    std::vector<int> fwd_degrees(g.num_nodes, 0);
    for (const auto& edge : remapped_edge_list) {
        fwd_degrees[edge.first]++;
    }
    for (int i = 1; i <= g.num_nodes; ++i) {
        g.forward_offsets[i] = g.forward_offsets[i - 1] + fwd_degrees[i - 1];
    }
    auto fwd_counters = g.forward_offsets;
    for (const auto& edge : remapped_edge_list) {
        g.forward_edges[fwd_counters[edge.first]++] = edge.second;
    }

    // Build backward graph (CSR)
    g.backward_offsets.assign(g.num_nodes + 1, 0);
    g.backward_edges.resize(g.num_edges);
    std::vector<int> bwd_degrees(g.num_nodes, 0);
    for (const auto& edge : remapped_edge_list) {
        bwd_degrees[edge.second]++;
    }
    for (int i = 1; i <= g.num_nodes; ++i) {
        g.backward_offsets[i] = g.backward_offsets[i - 1] + bwd_degrees[i - 1];
    }
    auto bwd_counters = g.backward_offsets;
    for (const auto& edge : remapped_edge_list) {
        g.backward_edges[bwd_counters[edge.second]++] = edge.first;
    }

    std::cout << "Graph loaded: " << g.num_nodes << " unique nodes, " << g.num_edges << " edges." << std::endl;
    return g;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << "p2p-Gnutella04.txt" << std::endl;
        return 1;
    }

    std::string filename = argv[1];
    Graph g = load_graph_from_file(filename);

    SccSolver solver(g);
    solver.run_method2();

    auto final_sccs = solver.get_sccs();

    std::cout << "\nFound " << final_sccs.size() << " Strongly Connected Components:" << std::endl;
    // Note: The output will be the remapped 0-indexed node IDs, not the original IDs from the file.
    // for (auto& scc : final_sccs) {
    //     std::sort(scc.begin(), scc.end());
    //     std::cout << "{ ";
    //     for (int node : scc) {
    //         std::cout << node << " ";
    //     }
    //     std::cout << "}" << std::endl;
    // }

    return 0;
}