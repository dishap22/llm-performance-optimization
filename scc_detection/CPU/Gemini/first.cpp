#include <iostream>
#include <vector>
#include <algorithm>
#include <atomic>
#include <deque>
#include <mutex>
#include <omp.h>

// Represents the graph using Compressed Sparse Row (CSR) format
// as recommended in Section 4.1 of the paper.
struct Graph {
    // Number of nodes
    int num_nodes;
    // Number of edges
    int num_edges;
    // Adjacency list for forward edges
    std::vector<int> forward_offsets;
    std::vector<int> forward_edges;
    // Adjacency list for backward (reverse) edges
    std::vector<int> backward_offsets;
    std::vector<int> backward_edges;
};

// Manages state and algorithms for SCC detection.
class SccSolver {
public:
    SccSolver(Graph& g) : graph(g) {
        // As per Section 3.1, 'color' and 'mark' auxiliary arrays are used
        // to avoid modifying the graph structure directly.
        colors.resize(g.num_nodes, 0);
        marks.resize(g.num_nodes, false);
        sccs.reserve(g.num_nodes);
        next_color = 1;
    }

    // Main entry point for Method 2 (Algorithm 9)
    void run_method2() {
        std::cout << "Phase 1: Parallel Trims, FWBW, and WCC detection" << std::endl;
        
        // Par-Trim' (initial)
        par_trim("Initial Par-Trim");

        // Par-FWBW (Algorithm 6, part of Method 1 and 2)
        // Detects the giant SCC using parallel traversal.
        par_fwbw();

        // Par-Trim' (Algorithm 9 calls for Par-Trim' which includes Trim2)
        // This is Par-Trim, Par-Trim2, then Par-Trim again.
        par_trim("Par-Trim post-FWBW");
        par_trim2();
        par_trim("Par-Trim post-Trim2");

        // Par-WCC (Algorithm 7)
        // Decomposes the remaining graph into weakly connected components.
        par_wcc();
        
        std::cout << "Phase 2: Parallelism in Recursion" << std::endl;
        // The work queue is now populated by par_wcc.
        // Process remaining components using the recursive FW-BW algorithm in parallel.
        #pragma omp parallel
        {
            process_work_queue();
        }

        std::cout << "SCC detection complete." << std::endl;
    }

    // Getter for the results
    const std::vector<std::vector<int>>& get_sccs() const {
        return sccs;
    }

private:
    Graph& graph;
    std::vector<int> colors;  // Corresponds to 'Color' array in the paper
    std::vector<bool> marks;  // Corresponds to 'mark' array in the paper
    std::vector<std::vector<int>> sccs;
    std::atomic<int> next_color;

    // Work queue for task-level parallelism (Section 4.3)
    std::deque<int> work_queue;
    std::mutex queue_mutex;


    // Implementation of Par-Trim (Algorithm 4)
    void par_trim(const std::string& phase_name) {
        bool changed;
        int iter = 0;
        do {
            changed = false;
            std::vector<int> new_sccs;

            #pragma omp parallel for
            for (int u = 0; u < graph.num_nodes; ++u) {
                if (!marks[u]) {
                    int in_degree = 0;
                    int out_degree = 0;

                    // Calculate in-degree within the same color partition
                    for (int i = graph.backward_offsets[u]; i < graph.backward_offsets[u+1]; ++i) {
                        int v = graph.backward_edges[i];
                        if (!marks[v] && colors[v] == colors[u]) {
                            in_degree++;
                        }
                    }

                    // Calculate out-degree within the same color partition
                    for (int i = graph.forward_offsets[u]; i < graph.forward_offsets[u+1]; ++i) {
                        int v = graph.forward_edges[i];
                        if (!marks[v] && colors[v] == colors[u]) {
                            out_degree++;
                        }
                    }
                    
                    if (in_degree == 0 || out_degree == 0) {
                        if (!marks[u]) { // Double check to avoid race condition
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
            }

            if(changed) {
                for(int node : new_sccs) {
                    sccs.push_back({node});
                }
            }
            iter++;
        } while (changed);
    }
    
    // Parallel Breadth-First Search for reachability
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
                    if (!marks[v] && colors[v] == target_color && visited[v] == 0) {
                        visited[v] = 1;
                        #pragma omp critical
                        next_frontier.push_back(v);
                    }
                }
            }
            frontier = next_frontier;
        }
    }

    // Parallel Forward-Backward step to find the giant SCC (Section 3.2)
    void par_fwbw() {
        int pivot = -1;
        // Pick a random unmarked node as pivot
        for(int i = 0; i < graph.num_nodes; ++i) {
            if(!marks[i]) {
                pivot = i;
                break;
            }
        }
        if (pivot == -1) return;

        std::vector<int> fw_visited(graph.num_nodes, 0);
        std::vector<int> bw_visited(graph.num_nodes, 0);

        // As suggested in section 4.2, use parallel BFS for Phase 1 traversals
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
        if (!scc_nodes.empty()) {
           sccs.push_back(scc_nodes);
        }
        work_queue.push_back(colors[pivot]); // Remainder
        work_queue.push_back(c_fw);
        work_queue.push_back(c_bw);
    }
    
    // Implementation of Par-Trim2 (Algorithm 8)
    void par_trim2() {
        std::vector<std::pair<int, int>> new_sccs_pairs;
        #pragma omp parallel for
        for (int u = 0; u < graph.num_nodes; ++u) {
            if (marks[u]) continue;

            // Pattern (a) from Figure 4
            if (get_degree(u, true, colors[u]) == 1 && get_degree(u, false, colors[u]) == 1) {
                int v = get_neighbor(u, true, colors[u]); // The only out-neighbor
                if (v != -1 && v == get_neighbor(u, false, colors[u])) { // v is also the only in-neighbor
                    if (get_degree(v, false, colors[v]) == 1) { // v has no other incoming edges
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
             // Pattern (b) from Figure 4
            if (!marks[u] && get_degree(u, true, colors[u]) == 1 && get_degree(u, false, colors[u]) == 1) {
                 int v = get_neighbor(u, true, colors[u]);
                 if (v != -1 && v == get_neighbor(u, false, colors[u])) {
                     if (get_degree(v, true, colors[v]) == 1) { // v has no other outgoing edges
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
        for(const auto& p : new_sccs_pairs) {
            sccs.push_back({p.first, p.second});
        }
    }
    
    // Implementation of Par-WCC (Algorithm 7)
    void par_wcc() {
        std::vector<int> wcc_ids(graph.num_nodes);
        #pragma omp parallel for
        for (int i = 0; i < graph.num_nodes; ++i) {
            wcc_ids[i] = i;
        }

        bool changed;
        do {
            changed = false;
            #pragma omp parallel for
            for (int u = 0; u < graph.num_nodes; ++u) {
                if (marks[u]) continue;
                // Propagate smallest ID among neighbors (undirected)
                for (int i = graph.forward_offsets[u]; i < graph.forward_offsets[u + 1]; ++i) {
                    int v = graph.forward_edges[i];
                    if (!marks[v] && colors[u] == colors[v]) {
                       if(wcc_ids[u] < wcc_ids[v]) {
                           #pragma omp atomic write
                           wcc_ids[v] = wcc_ids[u];
                           changed = true;
                       } else if(wcc_ids[v] < wcc_ids[u]) {
                           #pragma omp atomic write
                           wcc_ids[u] = wcc_ids[v];
                           changed = true;
                       }
                    }
                }
            }
        } while (changed);

        // Clear existing work queue and repopulate with WCCs
        std::lock_guard<std::mutex> lock(queue_mutex);
        work_queue.clear();
        
        #pragma omp parallel
        {
            std::vector<int> local_new_colors;
            #pragma omp for
            for (int i = 0; i < graph.num_nodes; ++i) {
                if (!marks[i] && wcc_ids[i] == i) { // This is a root of a WCC
                    int c = next_color++;
                    local_new_colors.push_back(c);
                    // Recolor all nodes in this component
                    for (int j = 0; j < graph.num_nodes; ++j) {
                        if (!marks[j] && wcc_ids[j] == i) {
                            colors[j] = c;
                        }
                    }
                }
            }
            #pragma omp critical
            for(int c : local_new_colors) {
                work_queue.push_back(c);
            }
        }
    }

    // Process tasks from the work queue (Section 4.3)
    void process_work_queue() {
        while (true) {
            int task_color = -1;
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (work_queue.empty()) {
                    break; 
                }
                task_color = work_queue.front();
                work_queue.pop_front();
            }

            if (task_color != -1) {
                recur_fwbw(task_color);
            }
        }
    }

    // Recursive Forward-Backward step for Phase 2 (Algorithm 5)
    void recur_fwbw(int color_c) {
        int pivot = -1;
        // Find a pivot in the current color partition
        for(int i = 0; i < graph.num_nodes; ++i) {
            if(!marks[i] && colors[i] == color_c) {
                pivot = i;
                break;
            }
        }

        if (pivot == -1) return; // Partition is empty

        std::vector<int> fw_reach, bw_reach;
        // Section 4.2 suggests using sequential DFS for the recursive phase
        dfs_reach(pivot, true, fw_reach, color_c);
        dfs_reach(pivot, false, bw_reach, color_c);

        std::vector<int> scc_nodes;
        std::sort(fw_reach.begin(), fw_reach.end());
        std::sort(bw_reach.begin(), bw_reach.end());
        std::set_intersection(fw_reach.begin(), fw_reach.end(),
                              bw_reach.begin(), bw_reach.end(),
                              std::back_inserter(scc_nodes));

        if (!scc_nodes.empty()) {
            int c_fw = next_color++;
            int c_bw = next_color++;
            
            for (int node : scc_nodes) {
                marks[node] = true;
            }

            // Repartition remaining nodes
            for (int node : fw_reach) if (!marks[node]) colors[node] = c_fw;
            for (int node : bw_reach) if (!marks[node]) colors[node] = c_bw;

            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                sccs.push_back(scc_nodes);
                // Push new subproblems to the queue
                work_queue.push_back(c_fw);
                work_queue.push_back(c_bw);
                work_queue.push_back(color_c); // Remainder
            }
        }
    }
    
    // Helper DFS for reachability used in recur_fwbw
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
    
    void dfs_reach(int start_node, bool forward, std::vector<int>& reach, int target_color) {
        std::vector<bool> visited(graph.num_nodes, false);
        dfs_reach(start_node, forward, reach, target_color, visited);
    }
    
    // Helpers for Par-Trim2
    int get_degree(int u, bool forward, int target_color) {
        int degree = 0;
        const auto& offsets = forward ? graph.forward_offsets : graph.backward_offsets;
        const auto& edges = forward ? graph.forward_edges : graph.backward_edges;
        for (int i = offsets[u]; i < offsets[u+1]; ++i) {
            int v = edges[i];
            if (!marks[v] && colors[v] == target_color) {
                degree++;
            }
        }
        return degree;
    }

    int get_neighbor(int u, bool forward, int target_color) {
        const auto& offsets = forward ? graph.forward_offsets : graph.backward_offsets;
        const auto& edges = forward ? graph.forward_edges : graph.backward_edges;
        for (int i = offsets[u]; i < offsets[u+1]; ++i) {
            int v = edges[i];
            if (!marks[v] && colors[v] == target_color) {
                return v;
            }
        }
        return -1;
    }
};


int main() {
    // Example usage with a simple graph
    // This graph has three SCCs: {0,1,2}, {3}, {4,5}
    Graph g;
    g.num_nodes = 6;
    g.num_edges = 8;

    // Edges: 0->1, 1->2, 2->0, 1->3, 3->4, 4->5, 5->4
    g.forward_offsets = {0, 1, 3, 4, 5, 7, 8};
    g.forward_edges = {1, 2, 3, 0, 4, 4, 5, 5};
    
    // Create backward edges for traversal
    g.backward_offsets.resize(g.num_nodes + 1, 0);
    g.backward_edges.resize(g.num_edges);
    std::vector<int> temp_counts(g.num_nodes, 0);
    for(int u=0; u<g.num_nodes; ++u) {
        for(int i=g.forward_offsets[u]; i<g.forward_offsets[u+1]; ++i) {
            int v = g.forward_edges[i];
            temp_counts[v]++;
        }
    }
    for(int i=1; i<=g.num_nodes; ++i) g.backward_offsets[i] = g.backward_offsets[i-1] + temp_counts[i-1];
    std::copy(g.backward_offsets.begin(), g.backward_offsets.end()-1, temp_counts.begin());
    for(int u=0; u<g.num_nodes; ++u) {
        for(int i=g.forward_offsets[u]; i<g.forward_offsets[u+1]; ++i) {
            int v = g.forward_edges[i];
            g.backward_edges[temp_counts[v]++] = u;
        }
    }


    SccSolver solver(g);
    solver.run_method2();

    auto final_sccs = solver.get_sccs();

    std::cout << "\nFound " << final_sccs.size() << " Strongly Connected Components:" << std::endl;
    for (const auto& scc : final_sccs) {
        std::cout << "{ ";
        for (int node : scc) {
            std::cout << node << " ";
        }
        std::cout << "}" << std::endl;
    }

    return 0;
}