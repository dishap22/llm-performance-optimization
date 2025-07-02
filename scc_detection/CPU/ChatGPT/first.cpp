// scc_method2.cpp
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

struct Graph {
    int V;
    vector<int> offset; // offset[i] is the starting index of edges from i
    vector<int> edges;  // edges list
    vector<int> in_deg, out_deg;

    Graph(int V) : V(V), offset(V+1, 0), in_deg(V, 0), out_deg(V, 0) {}

    void addEdge(int u, int v) {
        out_deg[u]++;
        in_deg[v]++;
    }

    void finalize() {
        vector<vector<int>> adj(V);
        for (int u = 0; u < V; ++u)
            adj[u].reserve(out_deg[u]);
        // Fill edges
        for (int u = 0; u < V; ++u) {
            for (int v : adj[u]) edges.push_back(v);
        }
    }
};

// Global structures
vector<int> color, mark, WCC;
vector<set<int>> SCCs;
int color_id = 1;
omp_lock_t lock;

void addSCC(const set<int>& scc) {
    omp_set_lock(&lock);
    SCCs.push_back(scc);
    omp_unset_lock(&lock);
}

void parTrim(Graph& g) {
    bool changed = true;
    while (changed) {
        changed = false;
        #pragma omp parallel for schedule(dynamic)
        for (int u = 0; u < g.V; ++u) {
            if (mark[u]) continue;
            if (g.in_deg[u] == 0 || g.out_deg[u] == 0) {
                mark[u] = 1;
                #pragma omp critical
                SCCs.emplace_back(set<int>{u});
                changed = true;
            }
        }
    }
}

void parTrim2(Graph& g) {
    #pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < g.V; ++u) {
        if (mark[u]) continue;
        // simulate in/out neighbors with example adjacency
        // Replace with real CSR traversal logic
        int in_neighbor = -1, out_neighbor = -1;
        // assume 1 in and 1 out for trim2 logic, check if reciprocal
        if (in_neighbor != -1 && out_neighbor == in_neighbor) {
            if (!mark[in_neighbor]) {
                mark[u] = mark[in_neighbor] = 1;
                #pragma omp critical
                SCCs.emplace_back(set<int>{u, in_neighbor});
            }
        }
    }
}

void BFS(const Graph& g, int start, vector<int>& reachable, bool forward) {
    queue<int> q;
    vector<int> visited(g.V, 0);
    q.push(start);
    visited[start] = 1;
    reachable.push_back(start);

    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int i = g.offset[u]; i < g.offset[u+1]; ++i) {
            int v = g.edges[i];
            if (!visited[v] && !mark[v]) {
                visited[v] = 1;
                reachable.push_back(v);
                q.push(v);
            }
        }
    }
}

void parFWBW(Graph& g, int seed) {
    vector<int> FW, BW;
    BFS(g, seed, FW, true);
    BFS(g, seed, BW, false);

    unordered_set<int> FWset(FW.begin(), FW.end());
    set<int> intersection;
    for (int v : BW) {
        if (FWset.count(v)) {
            mark[v] = 1;
            intersection.insert(v);
        }
    }
    if (!intersection.empty())
        addSCC(intersection);
}

void parWCC(Graph& g) {
    WCC.resize(g.V);
    #pragma omp parallel for
    for (int i = 0; i < g.V; ++i)
        WCC[i] = i;

    bool changed;
    do {
        changed = false;
        #pragma omp parallel for schedule(dynamic)
        for (int u = 0; u < g.V; ++u) {
            if (mark[u]) continue;
            for (int i = g.offset[u]; i < g.offset[u+1]; ++i) {
                int v = g.edges[i];
                if (WCC[v] < WCC[u]) {
                    WCC[u] = WCC[v];
                    changed = true;
                }
            }
        }
    } while (changed);
}

void recurFWBW(Graph& g, int seed) {
    if (mark[seed]) return;
    parFWBW(g, seed);

    // This would ideally create partitions and recurse.
    // For simplicity, here we simulate with one call.
}

void method2(Graph& g) {
    color.assign(g.V, 0);
    mark.assign(g.V, 0);
    omp_init_lock(&lock);

    // Phase 1
    parTrim(g);
    parFWBW(g, 0);
    parTrim2(g);
    parTrim(g);
    parWCC(g);

    // Phase 2 (parallel recursion)
    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < g.V; ++i) {
        if (!mark[i])
            recurFWBW(g, i);
    }

    omp_destroy_lock(&lock);
}

int main() {
    int V = 10;
    Graph g(V);
    // Example: add edges here
    // g.addEdge(u, v);

    g.finalize();
    method2(g);

    cout << "SCCs found: " << SCCs.size() << endl;
    for (auto& scc : SCCs) {
        for (int v : scc) cout << v << " ";
        cout << endl;
    }

    return 0;
}
