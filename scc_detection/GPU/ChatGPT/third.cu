// third.cu
// scc_cuda_hong_optimized.cu
// Compile with:
// nvcc -O3 -arch=sm_70 third.cu -o scc_hong_optimized

#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;

// CUDA error check
#define CUDA_CHECK(call)                                                    \
  do {                                                                      \
    cudaError_t err = call;                                                 \
    if (err != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
              cudaGetErrorString(err), __FILE__, __LINE__);                \
      exit(EXIT_FAILURE);                                                   \
    }                                                                       \
  } while (0)

// CSR loader
struct Graph {
    int N, E;
    vector<int> fwd_off, fwd_adj;
    vector<int> rev_off, rev_adj;
    unordered_map<int,int> reverse_map;
    Graph(const string &fname) { load(fname); }
private:
    void load(const string &fname) {
        ifstream in(fname);
        if (!in.is_open()) { cerr<<"Error opening "<<fname<<"\n"; exit(1); }
        vector<pair<int,int>> edges;
        unordered_map<int,int> mp;
        mp.reserve(1<<20);
        int nextId = 0;
        string line;
        while (getline(in, line)) {
            if (line.empty() || line[0]=='#') continue;
            istringstream ss(line);
            int u, v;
            if (!(ss >> u >> v)) continue;
            if (!mp.count(u)) { mp[u] = nextId; reverse_map[nextId] = u; nextId++; }
            if (!mp.count(v)) { mp[v] = nextId; reverse_map[nextId] = v; nextId++; }
            edges.emplace_back(mp[u], mp[v]);
        }
        N = nextId;
        E = edges.size();
        vector<vector<int>> fw(N), rv(N);
        for (auto &e: edges) {
            fw[e.first].push_back(e.second);
            rv[e.second].push_back(e.first);
        }
        fwd_off.resize(N+1);
        rev_off.resize(N+1);
        fwd_adj.resize(E);
        rev_adj.resize(E);
        fwd_off[0] = 0;
        rev_off[0] = 0;
        for (int i = 0; i < N; i++) {
            fwd_off[i+1] = fwd_off[i] + (int)fw[i].size();
            rev_off[i+1] = rev_off[i] + (int)rv[i].size();
        }
        vector<int> pf(N), pr(N);
        for (int i = 0; i < N; i++) {
            for (int v: fw[i]) fwd_adj[fwd_off[i] + pf[i]++] = v;
            for (int v: rv[i]) rev_adj[rev_off[i] + pr[i]++] = v;
        }
    }
};

// BFS expansion kernel
__global__ void bfs_expand(
    const int *off, const int *adj,
    int frontier_size,
    const int *in_frontier,
    int *out_frontier,
    int *out_count,
    int *visited)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= frontier_size) return;
    int u = in_frontier[tid];
    for (int e = off[u]; e < off[u+1]; ++e) {
        int v = adj[e];
        if (atomicExch(&visited[v], 1) == 0) {
            int pos = atomicAdd(out_count, 1);
            out_frontier[pos] = v;
        }
    }
}

// Intersection kernel
__global__ void collect_scc(
    int N,
    const int *fw_vis,
    const int *bw_vis,
    int *color,
    bool *mark,
    int curr_color)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u < N && !mark[u] && fw_vis[u] && bw_vis[u]) {
        mark[u] = true;
        color[u] = curr_color;
    }
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s graph.txt\n", argv[0]);
        return 1;
    }

    Graph G(argv[1]);
    int N = G.N, E = G.E;

    // Device memory
    int *d_fwd_off, *d_fwd_adj;
    int *d_rev_off, *d_rev_adj;
    int *d_front1, *d_front2;
    int *d_size1,  *d_size2;
    int *d_vis_fw, *d_vis_bw;
    int *d_color;
    bool *d_mark;

    CUDA_CHECK(cudaMalloc(&d_fwd_off,   (N+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_fwd_adj,    E   *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rev_off,   (N+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_rev_adj,    E   *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_front1,    N   *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_front2,    N   *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_size1,     sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_size2,     sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vis_fw,    N   *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_vis_bw,    N   *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_color,     N   *sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_mark,      N   *sizeof(bool)));

    // Copy CSR to device
    CUDA_CHECK(cudaMemcpy(d_fwd_off, G.fwd_off.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_fwd_adj, G.fwd_adj.data(),   E   *sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rev_off, G.rev_off.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rev_adj, G.rev_adj.data(),   E   *sizeof(int), cudaMemcpyHostToDevice));

    // Initialize mark/color
    CUDA_CHECK(cudaMemset(d_mark, 0, N*sizeof(bool)));
    CUDA_CHECK(cudaMemset(d_color,0, N*sizeof(int)));

    // Pinned host buffers
    int *h_frontier;
    int *h_count;
    CUDA_CHECK(cudaHostAlloc(&h_frontier, N*sizeof(int), cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_count,    sizeof(int),     cudaHostAllocPortable));

    // WCC segmentation (host-driven)
    vector<vector<int>> wccs;
    CUDA_CHECK(cudaMemset(d_vis_fw, 0, N*sizeof(int)));
    for (int p = 0; p < N; ++p) {
        CUDA_CHECK(cudaMemcpy(h_frontier, d_vis_fw, N*sizeof(int), cudaMemcpyDeviceToHost));
        if (h_frontier[p]) continue;
        int cnt = 1;
        CUDA_CHECK(cudaMemcpy(d_size1, &cnt, sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_front1, &p,   sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_vis_fw + p, 1, sizeof(int)));
        vector<int> grp{p};
        while (true) {
            CUDA_CHECK(cudaMemcpy(h_count, d_size1, sizeof(int), cudaMemcpyDeviceToHost));
            int fs = h_count[0];
            if (!fs) break;
            cnt = 0;
            CUDA_CHECK(cudaMemcpy(d_size2, &cnt, sizeof(int), cudaMemcpyHostToDevice));
            int gsz = (fs + 255) / 256;
            bfs_expand<<<gsz,256>>>(
                d_fwd_off, d_fwd_adj,
                fs,
                d_front1,
                d_front2,
                d_size2,
                d_vis_fw);
            CUDA_CHECK(cudaMemcpy(d_front1, d_front2, fs*sizeof(int), cudaMemcpyDeviceToDevice));
            swap(d_size1, d_size2);
            CUDA_CHECK(cudaMemcpy(h_frontier, d_front1, fs*sizeof(int), cudaMemcpyDeviceToHost));
            for (int i = 0; i < fs; ++i) grp.push_back(h_frontier[i]);
        }
        wccs.push_back(move(grp));
    }
    printf("%zu WCCs\n", wccs.size());

    // Per-WCC SCC detection
    for (size_t gi = 0; gi < wccs.size(); ++gi) {
        auto &grp = wccs[gi];
        printf("WCC %zu size=%zu\n", gi, grp.size());
        vector<char> host_mark(N, 0);
        CUDA_CHECK(cudaMemcpy(d_mark, host_mark.data(), N*sizeof(bool), cudaMemcpyHostToDevice));
        int scc_id = 1;
        for (int p : grp) {
            CUDA_CHECK(cudaMemcpy(h_frontier, d_mark, N*sizeof(bool), cudaMemcpyDeviceToHost));
            if (h_frontier[p]) continue;
            // forward BFS
            CUDA_CHECK(cudaMemset(d_vis_fw, 0, N*sizeof(int)));
            int cnt = 1;
            CUDA_CHECK(cudaMemcpy(d_size1, &cnt, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_front1, &p,   sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_vis_fw + p, 1, sizeof(int)));
            while (true) {
                CUDA_CHECK(cudaMemcpy(h_count, d_size1, sizeof(int), cudaMemcpyDeviceToHost));
                int fs = h_count[0];
                if (!fs) break;
                cnt = 0;
                CUDA_CHECK(cudaMemcpy(d_size2, &cnt, sizeof(int), cudaMemcpyHostToDevice));
                int gsz = (fs + 255) / 256;
                bfs_expand<<<gsz,256>>>(
                    d_fwd_off, d_fwd_adj,
                    fs,
                    d_front1,
                    d_front2,
                    d_size2,
                    d_vis_fw);
                swap(d_front1, d_front2);
                swap(d_size1, d_size2);
            }
            // backward BFS
            CUDA_CHECK(cudaMemset(d_vis_bw, 0, N*sizeof(int)));
            cnt = 1;
            CUDA_CHECK(cudaMemcpy(d_size1, &cnt, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_front1, &p,   sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_vis_bw + p, 1, sizeof(int)));
            while (true) {
                CUDA_CHECK(cudaMemcpy(h_count, d_size1, sizeof(int), cudaMemcpyDeviceToHost));
                int fs = h_count[0];
                if (!fs) break;
                cnt = 0;
                CUDA_CHECK(cudaMemcpy(d_size2, &cnt, sizeof(int), cudaMemcpyHostToDevice));
                int gsz = (fs + 255) / 256;
                bfs_expand<<<gsz,256>>>(
                    d_rev_off, d_rev_adj,
                    fs,
                    d_front1,
                    d_front2,
                    d_size2,
                    d_vis_bw);
                swap(d_front1, d_front2);
                swap(d_size1, d_size2);
            }
            // collect SCC
            int grid = (N + 255) / 256;
            collect_scc<<<grid,256>>>(N, d_vis_fw, d_vis_bw, d_color, d_mark, scc_id);
            CUDA_CHECK(cudaDeviceSynchronize());
            // optionally print
            printf("  SCC %d detected\n", scc_id);
            scc_id++;
        }
    }

    // Cleanup
    cudaFreeHost(h_frontier);
    cudaFreeHost(h_count);
    cudaFree(d_fwd_off);   cudaFree(d_fwd_adj);
    cudaFree(d_rev_off);   cudaFree(d_rev_adj);
    cudaFree(d_front1);    cudaFree(d_front2);
    cudaFree(d_size1);     cudaFree(d_size2);
    cudaFree(d_vis_fw);    cudaFree(d_vis_bw);
    cudaFree(d_color);     cudaFree(d_mark);
    return 0;
}