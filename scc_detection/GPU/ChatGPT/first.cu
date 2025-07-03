#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <queue>
#include <set>
#include <unordered_map>

#define THREADS_PER_BLOCK 512

struct Graph {
    int num_nodes;
    int num_edges;
    std::vector<int> row_offsets; // size num_nodes+1
    std::vector<int> col_indices; // size num_edges
};

__global__ void trim_kernel(int* row_offsets, int* col_indices, int* in_degrees, int* out_degrees, int* colors, int* marks, int num_nodes, bool* changed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes || marks[tid]) return;

    int out_deg = 0;
    int start = row_offsets[tid];
    int end = row_offsets[tid + 1];

    for (int i = start; i < end; ++i) {
        int dst = col_indices[i];
        if (!marks[dst]) out_deg++;
    }

    if (in_degrees[tid] == 0 || out_deg == 0) {
        marks[tid] = 1;
        colors[tid] = -1;
        *changed = true;
    }
}

__global__ void compute_degrees(int* row_offsets, int* col_indices, int* in_deg, int* out_deg, int* marks, int num_nodes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes || marks[tid]) return;

    out_deg[tid] = row_offsets[tid + 1] - row_offsets[tid];

    for (int i = row_offsets[tid]; i < row_offsets[tid + 1]; ++i) {
        int dst = col_indices[i];
        if (!marks[dst]) atomicAdd(&in_deg[dst], 1);
    }
}

__global__ void bfs_forward(int* row_offsets, int* col_indices, int* visited, int* colors, int num_nodes, int start) {
    __shared__ int frontier[THREADS_PER_BLOCK];
    __shared__ int next_frontier[THREADS_PER_BLOCK];
    int tid = threadIdx.x;

    if (tid == 0) {
        visited[start] = 1;
        colors[start] = 2;
    }
    __syncthreads();

    bool done = false;
    while (!done) {
        done = true;
        __syncthreads();
        if (tid < num_nodes && visited[tid] == 1) {
            visited[tid] = 2;
            int start_edge = row_offsets[tid];
            int end_edge = row_offsets[tid + 1];

            for (int i = start_edge; i < end_edge; ++i) {
                int dst = col_indices[i];
                if (visited[dst] == 0) {
                    visited[dst] = 1;
                    colors[dst] = 2;
                    done = false;
                }
            }
        }
        __syncthreads();
    }
}

void SCC_Detection_Method2(Graph& G) {
    int N = G.num_nodes;
    int M = G.num_edges;

    // Host data
    thrust::host_vector<int> h_colors(N, 0);
    thrust::host_vector<int> h_marks(N, 0);
    thrust::host_vector<int> h_in_deg(N, 0);
    thrust::host_vector<int> h_out_deg(N, 0);

    // Device data
    thrust::device_vector<int> d_row_offsets = G.row_offsets;
    thrust::device_vector<int> d_col_indices = G.col_indices;
    thrust::device_vector<int> d_colors = h_colors;
    thrust::device_vector<int> d_marks = h_marks;
    thrust::device_vector<int> d_in_deg(N, 0);
    thrust::device_vector<int> d_out_deg(N, 0);

    int* d_row_offsets_ptr = thrust::raw_pointer_cast(d_row_offsets.data());
    int* d_col_indices_ptr = thrust::raw_pointer_cast(d_col_indices.data());
    int* d_colors_ptr = thrust::raw_pointer_cast(d_colors.data());
    int* d_marks_ptr = thrust::raw_pointer_cast(d_marks.data());
    int* d_in_deg_ptr = thrust::raw_pointer_cast(d_in_deg.data());
    int* d_out_deg_ptr = thrust::raw_pointer_cast(d_out_deg.data());

    bool h_changed = true;
    bool* d_changed;
    cudaMalloc(&d_changed, sizeof(bool));

    // -------- Phase 1: Parallel Trimming (Trim1)
    while (h_changed) {
        cudaMemset(d_in_deg_ptr, 0, sizeof(int) * N);
        compute_degrees<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_row_offsets_ptr, d_col_indices_ptr, d_in_deg_ptr, d_out_deg_ptr, d_marks_ptr, N
        );
        cudaDeviceSynchronize();

        h_changed = false;
        cudaMemcpy(d_changed, &h_changed, sizeof(bool), cudaMemcpyHostToDevice);

        trim_kernel<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_row_offsets_ptr, d_col_indices_ptr, d_in_deg_ptr, d_out_deg_ptr, d_colors_ptr, d_marks_ptr, N, d_changed
        );
        cudaMemcpy(&h_changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    // -------- Phase 2: Forward Reachability (FWBW)
    int pivot = 0; // You can use random or better selection heuristics
    thrust::device_vector<int> d_visited(N, 0);
    bfs_forward<<<1, THREADS_PER_BLOCK>>>(
        d_row_offsets_ptr, d_col_indices_ptr, thrust::raw_pointer_cast(d_visited.data()), d_colors_ptr, N, pivot
    );
    cudaDeviceSynchronize();

    // NOTE: Skipping Trim2 and WCC steps for simplicity in this initial implementation
    // You can expand this by applying pattern matching and BFS for WCCs

    // Copy result back to host
    thrust::copy(d_colors.begin(), d_colors.end(), h_colors.begin());

    // -------- Output SCCs
    std::unordered_map<int, std::vector<int>> scc_map;
    for (int i = 0; i < N; ++i) {
        if (h_colors[i] >= 0) {
            scc_map[h_colors[i]].push_back(i);
        }
    }

    std::cout << "Number of SCCs: " << scc_map.size() << "\n";
    for (const auto& [color, nodes] : scc_map) {
        std::cout << "SCC: ";
        for (int node : nodes)
            std::cout << node << " ";
        std::cout << "\n";
    }

    cudaFree(d_changed);
}

int main() {
    // Sample graph: 0->1->2->0 and 3->4
    Graph G;
    G.num_nodes = 5;
    G.num_edges = 6;
    G.row_offsets = {0, 1, 2, 3, 4, 6};
    G.col_indices = {1, 2, 0, 4, 3, 3};

    SCC_Detection_Method2(G);
    return 0;
}
