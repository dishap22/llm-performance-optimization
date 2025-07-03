#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <unordered_map>
#include <stdexcept>
#include <deque>
#include <mutex>
#include <set>
#include <omp.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ===================================================================
// UTILITIES & FORWARD DECLARATIONS
// ===================================================================

// Macro for robust CUDA error checking
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        exit(1);
    }
}

// Structure to hold graph data on the host before moving to CUDA device
struct Graph {
    int num_nodes;
    int num_edges;
    std::vector<int> forward_offsets;
    std::vector<int> forward_edges;
    std::vector<int> backward_offsets;
    std::vector<int> backward_edges;
};

// Main class to manage the SCC detection process on the GPU
class SccSolverCuda {
public:
    SccSolverCuda(Graph& h_graph);
    ~SccSolverCuda();
    void run_method2();
    std::vector<std::vector<int>> get_sccs();

private:
    // Device pointers for graph structure
    int* d_forward_offsets;
    int* d_forward_edges;
    int* d_backward_offsets;
    int* d_backward_edges;

    // Device pointers for node states
    int* d_colors;
    int* d_marks;
    int* d_wcc_ids;
    int* d_scc_buffer;
    int* d_scc_buffer_count;

    // Host and device pointers for tracking changes
    int* h_changed;
    int* d_changed;

    // Graph dimensions
    int num_nodes;
    int num_edges;

    // Host-side data for managing the final phase
    Graph& host_graph;
    std::vector<std::vector<int>> final_sccs;
    int next_color;

    // GPU-heavy parallel functions
    void par_trim(const std::string& phase_name);
    void par_fwbw();
    void par_trim2();
    void par_wcc();

    // CPU-based processing for small, recursive tasks
    void process_work_queue_on_cpu(std::deque<int>& work_queue, std::vector<int>& h_colors, std::vector<int>& h_marks);
    void recur_fwbw_on_cpu(int color_c, std::vector<int>& h_colors, std::vector<int>& h_marks, std::deque<int>& work_queue, std::mutex& queue_mutex);
    void dfs_reach_on_cpu(int u, bool forward, std::vector<int>& reach, int target_color, const std::vector<int>& h_colors, const std::vector<int>& h_marks, std::vector<bool>& visited);

    // Helper to collect newly identified SCCs from the device
    void collect_sccs_from_buffer(const std::string& type);
};

// Function to load the graph from a file (CPU task)
Graph load_graph_from_file(const std::string& filename);

// ===================================================================
// KERNEL DEFINITIONS
// ===================================================================

__global__ void par_trim_kernel(int num_nodes, int* fwd_offsets, int* fwd_edges, int* bwd_offsets, int* bwd_edges, int* colors, int* marks, int* d_changed, int* d_scc_buffer, int* d_scc_buffer_count) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes || marks[u] == 1) return;

    int current_color = colors[u];
    int in_degree = 0;
    for (int i = bwd_offsets[u]; i < bwd_offsets[u + 1]; ++i) {
        int v = bwd_edges[i];
        if (marks[v] == 0 && colors[v] == current_color) in_degree++;
    }

    int out_degree = 0;
    for (int i = fwd_offsets[u]; i < fwd_offsets[u + 1]; ++i) {
        int v = fwd_edges[i];
        if (marks[v] == 0 && colors[v] == current_color) out_degree++;
    }

    if (in_degree == 0 || out_degree == 0) {
        if (atomicCAS(&marks[u], 0, 1) == 0) {
            atomicExch(d_changed, 1);
            int idx = atomicAdd(d_scc_buffer_count, 1);
            d_scc_buffer[idx] = u;
        }
    }
}

__global__ void par_trim2_kernel(int num_nodes, int* fwd_offsets, int* fwd_edges, int* bwd_offsets, int* bwd_edges, int* colors, int* marks, int* d_scc_buffer, int* d_scc_buffer_count) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes || marks[u] == 1) return;

    int current_color = colors[u];

    int fwd_degree = 0; int fwd_neighbor = -1;
    for (int i = fwd_offsets[u]; i < fwd_offsets[u + 1]; ++i) {
        int v_cand = fwd_edges[i];
        if (marks[v_cand] == 0 && colors[v_cand] == current_color) {
            fwd_degree++;
            fwd_neighbor = v_cand;
        }
    }

    int bwd_degree = 0; int bwd_neighbor = -1;
    for (int i = bwd_offsets[u]; i < bwd_offsets[u + 1]; ++i) {
        int v_cand = bwd_edges[i];
        if (marks[v_cand] == 0 && colors[v_cand] == current_color) {
            bwd_degree++;
            bwd_neighbor = v_cand;
        }
    }

    if (fwd_degree == 1 && bwd_degree == 1 && fwd_neighbor == bwd_neighbor) {
        int v = fwd_neighbor;
        if (u >= v) return;

        int v_bwd_degree = 0;
        for (int i = bwd_offsets[v]; i < bwd_offsets[v + 1]; ++i) {
            int w_cand = bwd_edges[i];
            if (marks[w_cand] == 0 && colors[w_cand] == current_color) v_bwd_degree++;
        }

        if (v_bwd_degree == 1) {
             if (atomicCAS(&marks[u], 0, 1) == 0 && atomicCAS(&marks[v], 0, 1) == 0) {
                int idx = atomicAdd(d_scc_buffer_count, 2);
                d_scc_buffer[idx] = u;
                d_scc_buffer[idx+1] = v;
            }
        }
    }
}

__global__ void wcc_init_kernel(int num_nodes, int* wcc_ids) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;
    wcc_ids[u] = u;
}

__global__ void wcc_propagate_kernel(int num_nodes, int* fwd_offsets, int* fwd_edges, int* bwd_offsets, int* bwd_edges, int* wcc_ids, int* colors, int* marks, int* d_changed) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes || marks[u] == 1) return;

    int u_wcc = wcc_ids[u];
    int current_color = colors[u];

    for (int i = fwd_offsets[u]; i < fwd_offsets[u + 1]; ++i) {
        int v = fwd_edges[i];
        if (marks[v] == 0 && colors[v] == current_color) {
            int v_wcc = wcc_ids[v];
            if (u_wcc < v_wcc) {
                if(atomicMin(&wcc_ids[v], u_wcc) > u_wcc) atomicExch(d_changed, 1);
            }
        }
    }
    for (int i = bwd_offsets[u]; i < bwd_offsets[u + 1]; ++i) {
        int v = bwd_edges[i];
        if (marks[v] == 0 && colors[v] == current_color) {
            int v_wcc = wcc_ids[v];
            if (u_wcc < v_wcc) {
                if(atomicMin(&wcc_ids[v], u_wcc) > u_wcc) atomicExch(d_changed, 1);
            }
        }
    }
}

__global__ void wcc_compress_kernel(int num_nodes, int* wcc_ids, int* marks) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes || marks[u] == 1) return;

    while(wcc_ids[u] != wcc_ids[wcc_ids[u]]) {
        wcc_ids[u] = wcc_ids[wcc_ids[u]];
    }
}

// ===================================================================
// SccSolverCuda Class Implementation
// ===================================================================

SccSolverCuda::SccSolverCuda(Graph& h_graph) : host_graph(h_graph) {
    num_nodes = h_graph.num_nodes;
    num_edges = h_graph.num_edges;
    next_color = 1;

    CHECK_CUDA_ERROR(cudaMalloc(&d_forward_offsets, (num_nodes + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_forward_edges, num_edges * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_backward_offsets, (num_nodes + 1) * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_backward_edges, num_edges * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_colors, num_nodes * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_marks, num_nodes * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_wcc_ids, num_nodes * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_scc_buffer, num_nodes * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_scc_buffer_count, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_changed, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMallocHost(&h_changed, sizeof(int)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_forward_offsets, h_graph.forward_offsets.data(), (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_forward_edges, h_graph.forward_edges.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_backward_offsets, h_graph.backward_offsets.data(), (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_backward_edges, h_graph.backward_edges.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMemset(d_colors, 0, num_nodes * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMemset(d_marks, 0, num_nodes * sizeof(int)));
}

SccSolverCuda::~SccSolverCuda() {
    cudaFree(d_forward_offsets);
    cudaFree(d_forward_edges);
    cudaFree(d_backward_offsets);
    cudaFree(d_backward_edges);
    cudaFree(d_colors);
    cudaFree(d_marks);
    cudaFree(d_wcc_ids);
    cudaFree(d_scc_buffer);
    cudaFree(d_scc_buffer_count);
    cudaFree(d_changed);
    cudaFreeHost(h_changed);
}

void SccSolverCuda::run_method2() {
    std::cout << "Phase 1: GPU Parallel Operations" << std::endl;
    par_trim("Initial Trim");
    par_fwbw(); // This is a placeholder as noted in the method
    par_trim("Post-FWBW Trim");
    par_trim2();
    par_trim("Post-Trim2 Trim");
    par_wcc();

    std::cout << "Phase 2: CPU Parallel Recursion" << std::endl;
    std::vector<int> h_colors(num_nodes);
    std::vector<int> h_marks(num_nodes);
    CHECK_CUDA_ERROR(cudaMemcpy(h_colors.data(), d_colors, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_marks.data(), d_marks, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));

    std::set<int> unique_colors;
    for(int i = 0; i < num_nodes; ++i) {
        if(h_marks[i] == 0) unique_colors.insert(h_colors[i]);
    }
    std::deque<int> work_queue(unique_colors.begin(), unique_colors.end());

    process_work_queue_on_cpu(work_queue, h_colors, h_marks);
}

void SccSolverCuda::par_trim(const std::string& phase_name) {
    int blocks = (num_nodes + 255) / 256;
    int threads = 256;
    std::cout << "--- " << phase_name << " ---" << std::endl;
    do {
        *h_changed = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(d_changed, h_changed, sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemset(d_scc_buffer_count, 0, sizeof(int)));

        par_trim_kernel<<<blocks, threads>>>(num_nodes, d_forward_offsets, d_forward_edges, d_backward_offsets, d_backward_edges, d_colors, d_marks, d_changed, d_scc_buffer, d_scc_buffer_count);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());

        collect_sccs_from_buffer("Trim");
        CHECK_CUDA_ERROR(cudaMemcpy(h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));

    } while (*h_changed == 1);
}

void SccSolverCuda::par_trim2() {
    std::cout << "--- Par-Trim2 ---" << std::endl;
    int blocks = (num_nodes + 255) / 256;
    int threads = 256;
    CHECK_CUDA_ERROR(cudaMemset(d_scc_buffer_count, 0, sizeof(int)));

    par_trim2_kernel<<<blocks, threads>>>(num_nodes, d_forward_offsets, d_forward_edges, d_backward_offsets, d_backward_edges, d_colors, d_marks, d_scc_buffer, d_scc_buffer_count);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    collect_sccs_from_buffer("Trim2");
}

void SccSolverCuda::par_fwbw() {
    std::cout << "--- Par-FWBW (Placeholder) ---" << std::endl;
    // A full implementation requires an efficient parallel BFS, which is complex.
    // This step would find the giant SCC, mark its nodes, and partition the rest of
    // the graph into FW and BW sets by updating d_colors.
}

void SccSolverCuda::par_wcc() {
    std::cout << "--- Par-WCC ---" << std::endl;
    int blocks = (num_nodes + 255) / 256;
    int threads = 256;

    wcc_init_kernel<<<blocks, threads>>>(num_nodes, d_wcc_ids);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    do {
        *h_changed = 0;
        CHECK_CUDA_ERROR(cudaMemcpy(d_changed, h_changed, sizeof(int), cudaMemcpyHostToDevice));
        wcc_propagate_kernel<<<blocks, threads>>>(num_nodes, d_forward_offsets, d_forward_edges, d_backward_offsets, d_backward_edges, d_wcc_ids, d_colors, d_marks, d_changed);
        CHECK_CUDA_ERROR(cudaDeviceSynchronize());
        CHECK_CUDA_ERROR(cudaMemcpy(h_changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost));
    } while (*h_changed == 1);

    for(int i=0; i<2; ++i) {
       wcc_compress_kernel<<<blocks, threads>>>(num_nodes, d_wcc_ids, d_marks);
       CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    }
}

void SccSolverCuda::process_work_queue_on_cpu(std::deque<int>& work_queue, std::vector<int>& h_colors, std::vector<int>& h_marks) {
    std::mutex queue_mutex;
    #pragma omp parallel
    {
        while(true) {
            int task_color = -1;
            {
                std::lock_guard<std::mutex> lock(queue_mutex);
                if (work_queue.empty()) break;
                task_color = work_queue.front();
                work_queue.pop_front();
            }
            if (task_color != -1) {
                recur_fwbw_on_cpu(task_color, h_colors, h_marks, work_queue, queue_mutex);
            }
        }
    }
}

void SccSolverCuda::recur_fwbw_on_cpu(int color_c, std::vector<int>& h_colors, std::vector<int>& h_marks, std::deque<int>& work_queue, std::mutex& queue_mutex) {
    int pivot = -1;
    for (int i = 0; i < num_nodes; ++i) {
        if (h_marks[i] == 0 && h_colors[i] == color_c) {
            pivot = i;
            break;
        }
    }
    if (pivot == -1) return;

    std::vector<int> fw_reach, bw_reach;
    std::vector<bool> visited_fw(num_nodes, false);
    dfs_reach_on_cpu(pivot, true, fw_reach, color_c, h_colors, h_marks, visited_fw);
    std::vector<bool> visited_bw(num_nodes, false);
    dfs_reach_on_cpu(pivot, false, bw_reach, color_c, h_colors, h_marks, visited_bw);

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

        #pragma omp critical
        {
            final_sccs.push_back(scc_nodes);
        }

        for (int node : scc_nodes) h_marks[node] = 1;

        std::vector<int> fw_only, bw_only;
        for (int node : fw_reach) if (h_marks[node] == 0) {
            h_colors[node] = c_fw;
            fw_only.push_back(node);
        }
        for (int node : bw_reach) if (h_marks[node] == 0 && fw_set.find(node) == fw_set.end()) {
            h_colors[node] = c_bw;
            bw_only.push_back(node);
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            if(!fw_only.empty()) work_queue.push_back(c_fw);
            if(!bw_only.empty()) work_queue.push_back(c_bw);
        }
    }
}

void SccSolverCuda::dfs_reach_on_cpu(int u, bool forward, std::vector<int>& reach, int target_color, const std::vector<int>& h_colors, const std::vector<int>& h_marks, std::vector<bool>& visited) {
    visited[u] = true;
    reach.push_back(u);
    const auto& offsets = forward ? host_graph.forward_offsets : host_graph.backward_offsets;
    const auto& edges = forward ? host_graph.forward_edges : host_graph.backward_edges;
    for (int i = offsets[u]; i < offsets[u+1]; ++i) {
        int v = edges[i];
        if (h_marks[v] == 0 && h_colors[v] == target_color && !visited[v]) {
            dfs_reach_on_cpu(v, forward, reach, target_color, h_colors, h_marks, visited);
        }
    }
}

void SccSolverCuda::collect_sccs_from_buffer(const std::string& type) {
    int count = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(&count, d_scc_buffer_count, sizeof(int), cudaMemcpyDeviceToHost));

    if (count > 0) {
        std::vector<int> h_scc_nodes(count);
        CHECK_CUDA_ERROR(cudaMemcpy(h_scc_nodes.data(), d_scc_buffer, count * sizeof(int), cudaMemcpyDeviceToHost));

        if (type == "Trim") {
            for(int node : h_scc_nodes) {
                final_sccs.push_back({node});
            }
        } else if (type == "Trim2") {
            for(size_t i = 0; i < h_scc_nodes.size(); i+=2) {
                final_sccs.push_back({h_scc_nodes[i], h_scc_nodes[i+1]});
            }
        }
    }
}

std::vector<std::vector<int>> SccSolverCuda::get_sccs() {
    return final_sccs;
}

// ===================================================================
// MAIN FUNCTION & GRAPH LOADING
// ===================================================================

Graph load_graph_from_file(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }
    std::vector<std::pair<int, int>> raw_edge_list;
    std::unordered_map<int, int> node_map;
    int next_node_id = 0;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        int u_orig, v_orig;
        if (ss >> u_orig >> v_orig) {
            if (node_map.find(u_orig) == node_map.end()) node_map[u_orig] = next_node_id++;
            if (node_map.find(v_orig) == node_map.end()) node_map[v_orig] = next_node_id++;
            raw_edge_list.push_back({node_map[u_orig], node_map[v_orig]});
        }
    }
    file.close();

    Graph g;
    g.num_nodes = node_map.size();
    g.num_edges = raw_edge_list.size();
    g.forward_offsets.assign(g.num_nodes + 1, 0);
    g.backward_offsets.assign(g.num_nodes + 1, 0);

    for (const auto& edge : raw_edge_list) {
        g.forward_offsets[edge.first + 1]++;
        g.backward_offsets[edge.second + 1]++;
    }
    for (int i = 1; i <= g.num_nodes; ++i) {
        g.forward_offsets[i] += g.forward_offsets[i - 1];
        g.backward_offsets[i] += g.backward_offsets[i - 1];
    }

    g.forward_edges.resize(g.num_edges);
    g.backward_edges.resize(g.num_edges);
    std::vector<int> fwd_counters = g.forward_offsets;
    std::vector<int> bwd_counters = g.backward_offsets;
    for (const auto& edge : raw_edge_list) {
        g.forward_edges[fwd_counters[edge.first]++] = edge.second;
        g.backward_edges[bwd_counters[edge.second]++] = edge.first;
    }
    std::cout << "Graph loaded: " << g.num_nodes << " nodes, " << g.num_edges << " edges." << std::endl;
    return g;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file.txt>" << std::endl;
        return 1;
    }

    Graph g = load_graph_from_file(argv[1]);

    SccSolverCuda solver(g);
    solver.run_method2();

    auto final_sccs = solver.get_sccs();
    std::cout << "\nFound " << final_sccs.size() << " Strongly Connected Components." << std::endl;
    // Uncomment to print all SCCs
    /*
    for (auto& scc : final_sccs) {
        std::sort(scc.begin(), scc.end());
        std::cout << "{ ";
        for (int node : scc) {
            std::cout << node << " ";
        }
        std::cout << "}" << std::endl;
    }
    */
    return 0;
}