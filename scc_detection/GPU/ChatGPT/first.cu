// scc_cuda_kosaraju.cu
#include <bits/stdc++.h>
#include <cuda_runtime.h>
using namespace std;

//---------------------------------------------
// CUDA error check
#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    cudaError_t err = call;                                                     \
    if (err != cudaSuccess) {                                                   \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                               \
              cudaGetErrorString(err), __FILE__, __LINE__);                    \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

//---------------------------------------------
// Host CSR graph
struct Graph {
  int N, E;
  vector<int> fwd_off, fwd_adj;
  vector<int> rev_off, rev_adj;
  unordered_map<int,int> reverse_map;  // map from 0..N-1 back to original IDs

  Graph(const string &fname) { load(fname); }

private:
  void load(const string &fname) {
    ifstream in(fname);
    if (!in.is_open()) { cerr<<"Error opening "<<fname<<"\n"; exit(1); }

    vector<pair<int,int>> edges;
    unordered_map<int,int> mp;
    mp.reserve(1<<20);
    int nextId=0;
    string line;
    while (getline(in,line)) {
      if (line.empty()||line[0]=='#') continue;
      istringstream ss(line);
      int u,v;
      if (!(ss>>u>>v)) continue;
      if (!mp.count(u)) { mp[u]=nextId; reverse_map[nextId]=u; nextId++; }
      if (!mp.count(v)) { mp[v]=nextId; reverse_map[nextId]=v; nextId++; }
      edges.emplace_back(mp[u], mp[v]);
    }

    N = nextId;
    E = edges.size();
    vector<vector<int>> fw(N), rv(N);
    for (auto &e: edges) {
      fw[e.first].push_back(e.second);
      rv[e.second].push_back(e.first);
    }

    fwd_off.resize(N+1); rev_off.resize(N+1);
    fwd_adj.resize(E);   rev_adj.resize(E);
    fwd_off[0]=0; rev_off[0]=0;
    for (int i=0;i<N;i++){
      fwd_off[i+1] = fwd_off[i] + fw[i].size();
      rev_off[i+1] = rev_off[i] + rv[i].size();
    }
    vector<int> pf(N,0), pr(N,0);
    for (int i=0;i<N;i++){
      for (int v: fw[i]) fwd_adj[fwd_off[i] + pf[i]++] = v;
      for (int v: rv[i]) rev_adj[rev_off[i] + pr[i]++] = v;
    }
  }
};

//---------------------------------------------
// Device buffers
int  *d_fwd_off, *d_fwd_adj,
     *d_rev_off, *d_rev_adj,
     *d_frontier_curr, *d_frontier_next,
     *d_frontier_size_curr, *d_frontier_size_next,
     *d_visited_fw, *d_visited_bw,
     *d_color;
bool *d_mark;

//---------------------------------------------
// Kernel: BFS expand one frontier
__global__ void bfs_expand(
    const int *off, const int *adj,
    int frontier_size,
    const int *frontier_curr,
    int *frontier_next,
    int *next_size,
    int *visited)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  if (idx >= frontier_size) return;
  int u = frontier_curr[idx];
  for (int e = off[u]; e < off[u+1]; ++e) {
    int v = adj[e];
    if (atomicExch(&visited[v], 1) == 0) {
      int pos = atomicAdd(next_size, 1);
      frontier_next[pos] = v;
    }
  }
}

//---------------------------------------------
// Kernel: collect intersection = an SCC
__global__ void collect_scc(
    int N,
    const int *visited_fw,
    const int *visited_bw,
    int *color,
    bool *mark,
    int curr_color)
{
  int u = blockIdx.x*blockDim.x + threadIdx.x;
  if (u >= N) return;
  if (!mark[u] && visited_fw[u] && visited_bw[u]) {
    mark[u]  = true;
    color[u] = curr_color;
  }
}

//---------------------------------------------
// Host: run SCC via forward-backward per pivot
int run_scc(const Graph &G) {
  int N = G.N;
  // copy CSR to device
  CUDA_CHECK(cudaMalloc(&d_fwd_off, (N+1)*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_fwd_adj,  G.E *sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_rev_off, (N+1)*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_rev_adj,  G.E *sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_fwd_off, G.fwd_off.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_fwd_adj, G.fwd_adj.data(),  G.E *sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rev_off, G.rev_off.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rev_adj, G.rev_adj.data(),  G.E *sizeof(int), cudaMemcpyHostToDevice));

  // alloc other buffers
  CUDA_CHECK(cudaMalloc(&d_frontier_curr,   N*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_frontier_next,   N*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_frontier_size_curr, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_frontier_size_next, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_visited_fw,  N*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_visited_bw,  N*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_color,       N*sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_mark,        N*sizeof(bool)));

  CUDA_CHECK(cudaMemset(d_mark,  0, N*sizeof(bool)));
  CUDA_CHECK(cudaMemset(d_color, 0, N*sizeof(int)));

  int h_color = 1;
  vector<int> h_fs(1), h_zero(1);
  vector<char> h_mark(N);

  dim3 block(256), grid((N+255)/256);

  while (true) {
    // find pivot
    CUDA_CHECK(cudaMemcpy(h_mark.data(), d_mark, N*sizeof(bool), cudaMemcpyDeviceToHost));
    int pivot=-1;
    for (int i=0;i<N;i++) if (!h_mark[i]) { pivot=i; break; }
    if (pivot<0) break;

    // --- forward BFS ---
    CUDA_CHECK(cudaMemset(d_visited_fw, 0, N*sizeof(int)));
    h_fs[0]=1;
    CUDA_CHECK(cudaMemcpy(d_frontier_size_curr, h_fs.data(), sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier_curr, &pivot, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_visited_fw + pivot, 1, sizeof(int)));

    while (true) {
      h_zero[0]=0;
      CUDA_CHECK(cudaMemcpy(d_frontier_size_next, h_zero.data(), sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(h_fs.data(), d_frontier_size_curr, sizeof(int), cudaMemcpyDeviceToHost));
      int fs = h_fs[0];
      if (fs==0) break;
      int gsz=(fs+255)/256;
      bfs_expand<<<gsz,256>>>(d_fwd_off, d_fwd_adj, fs,
                              d_frontier_curr,
                              d_frontier_next, d_frontier_size_next,
                              d_visited_fw);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(d_frontier_size_curr, d_frontier_size_next, sizeof(int), cudaMemcpyDeviceToDevice));
      swap(d_frontier_curr, d_frontier_next);
    }

    // --- backward BFS ---
    CUDA_CHECK(cudaMemset(d_visited_bw, 0, N*sizeof(int)));
    h_fs[0]=1;
    CUDA_CHECK(cudaMemcpy(d_frontier_size_curr, h_fs.data(), sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_frontier_curr, &pivot, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_visited_bw + pivot, 1, sizeof(int)));

    while (true) {
      h_zero[0]=0;
      CUDA_CHECK(cudaMemcpy(d_frontier_size_next, h_zero.data(), sizeof(int), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(h_fs.data(), d_frontier_size_curr, sizeof(int), cudaMemcpyDeviceToHost));
      int fs = h_fs[0];
      if (fs==0) break;
      int gsz=(fs+255)/256;
      bfs_expand<<<gsz,256>>>(d_rev_off, d_rev_adj, fs,
                              d_frontier_curr,
                              d_frontier_next, d_frontier_size_next,
                              d_visited_bw);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(d_frontier_size_curr, d_frontier_size_next, sizeof(int), cudaMemcpyDeviceToDevice));
      swap(d_frontier_curr, d_frontier_next);
    }

    // collect & print this SCC
    collect_scc<<<grid,block>>>(N,
                                d_visited_fw,
                                d_visited_bw,
                                d_color,
                                d_mark,
                                h_color);
    CUDA_CHECK(cudaDeviceSynchronize());

    // copy back colors for printing
    vector<int> h_col(N);
    CUDA_CHECK(cudaMemcpy(h_col.data(), d_color, N*sizeof(int), cudaMemcpyDeviceToHost));
    // cout<<"SCC "<<h_color<<": ";
    // for (int i=0;i<N;i++){
    //   if (h_col[i]==h_color) {
    //     cout<< G.reverse_map.at(i) <<" ";
    //   }
    // }
    // cout<<"\n";

    h_color++;
  }

  return h_color - 1;
}

//---------------------------------------------
int main(int argc, char** argv) {
  if (argc!=2) { cerr<<"Usage: "<<argv[0]<<" graph.txt\n"; return 1; }

  Graph G(argv[1]);

  // everything else is in run_scc
  auto t0 = chrono::high_resolution_clock::now();
  int total = run_scc(G);
  auto t1 = chrono::high_resolution_clock::now();

  cout<<"\nTotal Strongly Connected Components: "<<total<<"\n";
  double ms = chrono::duration<double, milli>(t1 - t0).count();
  cout<<"Total Execution Time: "<<ms<<" ms\n";
  return 0;
}