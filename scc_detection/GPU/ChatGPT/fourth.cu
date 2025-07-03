// CUDA version of the SCC Solver based on Method 2 (Trim, Trim2, WCC, FWBW)
// Graph is read from a SNAP-style .txt file

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <set>
#include <queue>

#define THREADS_PER_BLOCK 512

struct Graph {
    int num_nodes;
    int num_edges;
    std::vector<int> row_offsets;
    std::vector<int> col_indices;
};

__global__ void compute_degrees(int* row_offsets, int* col_indices, int* in_deg, int* out_deg, int* marks, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N || marks[tid]) return;
    out_deg[tid] = row_offsets[tid + 1] - row_offsets[tid];
    for (int i = row_offsets[tid]; i < row_offsets[tid + 1]; ++i) {
        int v = col_indices[i];
        if (!marks[v]) atomicAdd(&in_deg[v], 1);
    }
}

__global__ void trim_kernel(int* in_deg, int* out_deg, int* marks, int* colors, int N, bool* changed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N || marks[tid]) return;
    if (in_deg[tid] == 0 || out_deg[tid] == 0) {
        marks[tid] = 1;
        colors[tid] = -1;
        *changed = true;
    }
}

__global__ void wcc_label_propagation(int* row_offsets, int* col_indices, int* labels, int* marks, bool* changed, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N || marks[tid]) return;
    int my_label = labels[tid];
    for (int i = row_offsets[tid]; i < row_offsets[tid + 1]; ++i) {
        int nbr = col_indices[i];
        if (marks[nbr]) continue;
        if (labels[nbr] > my_label) {
            atomicMin(&labels[nbr], my_label);
            *changed = true;
        } else if (labels[nbr] < my_label) {
            atomicMin(&labels[tid], labels[nbr]);
            *changed = true;
        }
    }
}

__global__ void bfs_kernel(int* row_offsets, int* col_indices, int* active, int* visited, int* marks, int* labels, int label, int N, bool* changed, bool forward) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N || !active[tid] || visited[tid]) return;
    visited[tid] = 1;
    active[tid] = 0;

    if (forward) {
        for (int i = row_offsets[tid]; i < row_offsets[tid + 1]; ++i) {
            int nbr = col_indices[i];
            if (!marks[nbr] && labels[nbr] == label && !visited[nbr]) {
                active[nbr] = 1;
                *changed = true;
            }
        }
    } else {
        for (int u = 0; u < N; ++u) {
            for (int i = row_offsets[u]; i < row_offsets[u + 1]; ++i) {
                if (col_indices[i] == tid && !marks[u] && labels[u] == label && !visited[u]) {
                    active[u] = 1;
                    *changed = true;
                }
            }
        }
    }
}

Graph load_graph_from_file(const std::string& filename) {
    std::ifstream infile(filename);
    std::vector<std::pair<int, int>> edges;
    int max_node = 0;
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
        int idx = row_offsets[e.first] + counter[e.first]++;
        col_indices[idx] = e.second;
    }
    return Graph{n, static_cast<int>(edges.size()), row_offsets, col_indices};
}

void SCC_Method2_CUDA(Graph& G) {
    int N = G.num_nodes;

    int *d_row_offsets, *d_col_indices, *d_in_deg, *d_out_deg, *d_marks, *d_colors, *d_labels;
    bool *d_changed;
    cudaMalloc(&d_row_offsets, sizeof(int) * G.row_offsets.size());
    cudaMalloc(&d_col_indices, sizeof(int) * G.col_indices.size());
    cudaMalloc(&d_in_deg, sizeof(int) * N);
    cudaMalloc(&d_out_deg, sizeof(int) * N);
    cudaMalloc(&d_marks, sizeof(int) * N);
    cudaMalloc(&d_colors, sizeof(int) * N);
    cudaMalloc(&d_labels, sizeof(int) * N);
    cudaMalloc(&d_changed, sizeof(bool));

    cudaMemcpy(d_row_offsets, G.row_offsets.data(), sizeof(int) * G.row_offsets.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, G.col_indices.data(), sizeof(int) * G.col_indices.size(), cudaMemcpyHostToDevice);
    cudaMemset(d_marks, 0, sizeof(int) * N);
    cudaMemset(d_colors, 0, sizeof(int) * N);

    thrust::device_vector<int> d_labels_vec(N);
    thrust::sequence(d_labels_vec.begin(), d_labels_vec.end());
    cudaMemcpy(d_labels, thrust::raw_pointer_cast(d_labels_vec.data()), sizeof(int) * N, cudaMemcpyDeviceToDevice);

    bool h_changed = true;
    while (h_changed) {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
        cudaMemset(d_in_deg, 0, sizeof(int) * N);
        cudaMemset(d_out_deg, 0, sizeof(int) * N);
        compute_degrees<<<(N + 511)/512, 512>>>(d_row_offsets, d_col_indices, d_in_deg, d_out_deg, d_marks, N);
        trim_kernel<<<(N + 511)/512, 512>>>(d_in_deg, d_out_deg, d_marks, d_colors, N, d_changed);
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    // WCC
    h_changed = true;
    while (h_changed) {
        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
        wcc_label_propagation<<<(N + 511)/512, 512>>>(d_row_offsets, d_col_indices, d_labels, d_marks, d_changed, N);
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    thrust::host_vector<int> h_labels(N);
    cudaMemcpy(h_labels.data(), d_labels, sizeof(int) * N, cudaMemcpyDeviceToHost);

    std::unordered_map<int, std::vector<int>> label_groups;
    for (int i = 0; i < N; ++i) label_groups[h_labels[i]].push_back(i);

    thrust::device_vector<int> d_active(N);
    thrust::device_vector<int> d_fw(N);
    thrust::device_vector<int> d_bw(N);
    std::set<std::set<int>> scc_set;

    for (const auto& [label, group] : label_groups) {
        thrust::fill(d_active.begin(), d_active.end(), 0);
        thrust::fill(d_fw.begin(), d_fw.end(), 0);
        thrust::fill(d_bw.begin(), d_bw.end(), 0);

        d_active[group[0]] = 1;
        h_changed = true;
        while (h_changed) {
            h_changed = false;
            cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
            bfs_kernel<<<(N + 511)/512, 512>>>(d_row_offsets, d_col_indices,
                thrust::raw_pointer_cast(d_active.data()),
                thrust::raw_pointer_cast(d_fw.data()),
                d_marks, d_labels, label, N, d_changed, true);
            cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        }

        d_active[group[0]] = 1;
        h_changed = true;
        while (h_changed) {
            h_changed = false;
            cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);
            bfs_kernel<<<(N + 511)/512, 512>>>(d_row_offsets, d_col_indices,
                thrust::raw_pointer_cast(d_active.data()),
                thrust::raw_pointer_cast(d_bw.data()),
                d_marks, d_labels, label, N, d_changed, false);
            cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
        }

        thrust::host_vector<int> h_fw = d_fw;
        thrust::host_vector<int> h_bw = d_bw;
        std::set<int> scc;
        for (int i = 0; i < N; ++i) if (h_fw[i] && h_bw[i]) scc.insert(i);
        if (!scc.empty()) scc_set.insert(scc);
    }

    std::cout << "\nTotal SCCs found: " << scc_set.size() << "\n";


    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_in_deg);
    cudaFree(d_out_deg);
    cudaFree(d_marks);
    cudaFree(d_colors);
    cudaFree(d_labels);
    cudaFree(d_changed);
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph_file.txt>\n";
        return 1;
    }
    std::string filename = argv[1];
    Graph G = load_graph_from_file(filename);
    std::cout << "Graph loaded with " << G.num_nodes << " nodes and " << G.num_edges << " edges.\n";
    SCC_Method2_CUDA(G);
    return 0;
}
