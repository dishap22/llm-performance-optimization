// SCC detection using Method 2 in CUDA
// Input from .txt file
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <set>
#include <fstream>
#include <sstream>
#include <algorithm>

#define THREADS_PER_BLOCK 512

struct Graph {
    int num_nodes;
    int num_edges;
    std::vector<int> row_offsets; // CSR
    std::vector<int> col_indices;
};

Graph load_graph_from_txt(const std::string& filename) {
    std::ifstream infile(filename);
    int max_node = 0;
    std::vector<std::pair<int, int>> edges;
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        int u, v;
        if (iss >> u >> v) {
            edges.emplace_back(u, v);
            max_node = std::max({max_node, u, v});
        }
    }
    int n = max_node + 1;
    std::vector<int> out_deg(n, 0);
    for (auto& e : edges) out_deg[e.first]++;
    std::vector<int> row_offsets(n + 1, 0);
    for (int i = 0; i < n; ++i) row_offsets[i + 1] = row_offsets[i] + out_deg[i];
    std::vector<int> col_indices(edges.size());
    std::vector<int> counter(n, 0);
    for (auto& e : edges) {
        int u = e.first;
        int idx = row_offsets[u] + counter[u]++;
        col_indices[idx] = e.second;
    }
    return Graph{n, static_cast<int>(edges.size()), row_offsets, col_indices};
}

__global__ void trim_kernel(int* row_offsets, int* col_indices, int* in_deg, int* out_deg, int* color, int* mark, int N, bool* changed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N || mark[tid]) return;
    if (in_deg[tid] == 0 || out_deg[tid] == 0) {
        mark[tid] = 1;
        color[tid] = -1;
        *changed = true;
    }
}

__global__ void compute_degrees(int* row_offsets, int* col_indices, int* in_deg, int* out_deg, int* mark, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N || mark[tid]) return;
    out_deg[tid] = row_offsets[tid + 1] - row_offsets[tid];
    for (int i = row_offsets[tid]; i < row_offsets[tid + 1]; ++i) {
        int dst = col_indices[i];
        if (!mark[dst]) atomicAdd(&in_deg[dst], 1);
    }
}

__global__ void trim2_kernel(int* row_offsets, int* col_indices, int* in_deg, int* out_deg, int* color, int* mark, int N, bool* changed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N || mark[tid]) return;
    if (in_deg[tid] == 1) {
        int start = row_offsets[tid];
        int end = row_offsets[tid + 1];
        for (int i = start; i < end; ++i) {
            int dst = col_indices[i];
            if (out_deg[dst] == 1 && !mark[dst]) {
                color[tid] = color[dst] = -1;
                mark[tid] = mark[dst] = 1;
                *changed = true;
                break;
            }
        }
    }
}

__global__ void bfs_color(int* row_offsets, int* col_indices, int* mark, int* color, int N, int current_color) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N || mark[tid] || color[tid] != 0) return;
    
    color[tid] = current_color;
    for (int i = row_offsets[tid]; i < row_offsets[tid + 1]; ++i) {
        int dst = col_indices[i];
        if (!mark[dst] && color[dst] == 0) {
            color[dst] = current_color;
        }
    }
}

void scc_recursive_fwbw(Graph& G, thrust::device_vector<int>& d_row_offsets, thrust::device_vector<int>& d_col_indices,
                        thrust::device_vector<int>& d_color, thrust::device_vector<int>& d_mark, int current_color) {
    int N = G.num_nodes;
    thrust::host_vector<int> h_color = d_color;
    int pivot = -1;
    for (int i = 0; i < N; ++i) {
        if (h_color[i] == current_color && !d_mark[i]) {
            pivot = i;
            break;
        }
    }
    if (pivot == -1) return;

    // Forward reach
    thrust::device_vector<int> d_fw(N, 0);
    d_fw[pivot] = 1;
    thrust::host_vector<int> changed(N);
    bool flag = true;
    while (flag) {
        flag = false;
        thrust::copy(d_fw.begin(), d_fw.end(), changed.begin());
        for (int tid = 0; tid < N; ++tid) {
            if (changed[tid] == 1 && h_color[tid] == current_color && !d_mark[tid]) {
                for (int i = G.row_offsets[tid]; i < G.row_offsets[tid + 1]; ++i) {
                    int dst = G.col_indices[i];
                    if (h_color[dst] == current_color && !d_mark[dst] && d_fw[dst] == 0) {
                        d_fw[dst] = 1;
                        flag = true;
                    }
                }
            }
        }
    }

    // Backward reach
    thrust::device_vector<int> d_bw(N, 0);
    d_bw[pivot] = 1;
    flag = true;
    while (flag) {
        flag = false;
        thrust::copy(d_bw.begin(), d_bw.end(), changed.begin());
        for (int tid = 0; tid < N; ++tid) {
            if (changed[tid] == 1 && h_color[tid] == current_color && !d_mark[tid]) {
                for (int i = 0; i < G.num_nodes; ++i) {
                    for (int j = G.row_offsets[i]; j < G.row_offsets[i + 1]; ++j) {
                        if (G.col_indices[j] == tid && h_color[i] == current_color && !d_mark[i] && d_bw[i] == 0) {
                            d_bw[i] = 1;
                            flag = true;
                        }
                    }
                }
            }
        }
    }

    // Intersection = SCC
    thrust::host_vector<int> fw = d_fw;
    thrust::host_vector<int> bw = d_bw;
    for (int i = 0; i < N; ++i) {
        if (fw[i] && bw[i]) {
            d_mark[i] = 1;
            d_color[i] = -2; // Mark as part of this SCC
        }
    }

    // Recurse on FW-BW difference
    for (int i = 0; i < N; ++i) {
        if (fw[i] && !bw[i]) d_color[i] = current_color + 1;
        else if (!fw[i] && bw[i]) d_color[i] = current_color + 2;
    }
    scc_recursive_fwbw(G, d_row_offsets, d_col_indices, d_color, d_mark, current_color + 1);
    scc_recursive_fwbw(G, d_row_offsets, d_col_indices, d_color, d_mark, current_color + 2);
}

void SCC_Method2(Graph& G) {
    int N = G.num_nodes;
    int M = G.num_edges;

    thrust::device_vector<int> d_row_offsets = G.row_offsets;
    thrust::device_vector<int> d_col_indices = G.col_indices;
    thrust::device_vector<int> d_color(N, 0);
    thrust::device_vector<int> d_mark(N, 0);
    thrust::device_vector<int> d_in_deg(N, 0);
    thrust::device_vector<int> d_out_deg(N, 0);

    bool h_changed = true;
    bool* d_changed;
    cudaMalloc(&d_changed, sizeof(bool));

    // Phase 1: Trim
    while (h_changed) {
        cudaMemset(thrust::raw_pointer_cast(d_in_deg.data()), 0, sizeof(int) * N);
        compute_degrees<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_row_offsets.data()),
            thrust::raw_pointer_cast(d_col_indices.data()),
            thrust::raw_pointer_cast(d_in_deg.data()),
            thrust::raw_pointer_cast(d_out_deg.data()),
            thrust::raw_pointer_cast(d_mark.data()), N);
        cudaDeviceSynchronize();

        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        trim_kernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            thrust::raw_pointer_cast(d_row_offsets.data()),
            thrust::raw_pointer_cast(d_col_indices.data()),
            thrust::raw_pointer_cast(d_in_deg.data()),
            thrust::raw_pointer_cast(d_out_deg.data()),
            thrust::raw_pointer_cast(d_color.data()),
            thrust::raw_pointer_cast(d_mark.data()), N, d_changed);
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    // Trim2
    h_changed = false;
    cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
    trim2_kernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_row_offsets.data()),
        thrust::raw_pointer_cast(d_col_indices.data()),
        thrust::raw_pointer_cast(d_in_deg.data()),
        thrust::raw_pointer_cast(d_out_deg.data()),
        thrust::raw_pointer_cast(d_color.data()),
        thrust::raw_pointer_cast(d_mark.data()), N, d_changed);

    // WCC coloring
    int current_color = 1;
    bfs_color<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
        thrust::raw_pointer_cast(d_row_offsets.data()),
        thrust::raw_pointer_cast(d_col_indices.data()),
        thrust::raw_pointer_cast(d_mark.data()),
        thrust::raw_pointer_cast(d_color.data()), N, current_color);
    cudaDeviceSynchronize();

    // Recursive FWBW
    scc_recursive_fwbw(G, d_row_offsets, d_col_indices, d_color, d_mark, current_color);

    // Output SCCs
    thrust::host_vector<int> h_color = d_color;
    std::unordered_map<int, std::set<int>> scc_map;
    for (int i = 0; i < N; ++i) {
        if (h_color[i] < 0) scc_map[h_color[i]].insert(i);
    }
    std::cout << "Number of SCCs: " << scc_map.size() << "\n";
    for (const auto& [cid, nodes] : scc_map) {
        std::cout << "SCC: ";
        for (int v : nodes) std::cout << v << " ";
        std::cout << "\n";
    }
    cudaFree(d_changed);
}

int main() {

    Graph G = load_graph_from_txt("graph.txt");
    SCC_Method2(G);
    return 0;
}