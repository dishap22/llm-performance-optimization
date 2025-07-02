// scc_method2.cpp
#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

struct Graph {
    int V;
    vector<int> offset; // offset[i] = start of adjacency list for node i
    vector<int> edges;  // flattened adjacency lists
    vector<int> in_deg, out_deg;

    Graph(int V) : V(V), in_deg(V, 0), out_deg(V, 0) {}

    void buildFromEdges(const vector<pair<int, int>>& edgeList) {
        vector<vector<int>> adj(V);
        for (auto& [u, v] : edgeList) {
            adj[u].push_back(v);
            out_deg[u]++;
            in_deg[v]++;
        }
        offset.resize(V + 1);
        for (int i = 0; i < V; ++i)
            offset[i + 1] = offset[i] + adj[i].size();
        edges.resize(offset[V]);
        #pragma omp parallel for
        for (int i = 0; i < V; ++i)
            copy(adj[i].begin(), adj[i].end(), edges.begin() + offset[i]);
    }
};

vector<int> color, mark, WCC;
vector<set<int>> SCCs;
int color_id = 1;
omp_lock_t scc_lock;

void addSCC(const set<int>& scc) {
    omp_set_lock(&scc_lock);
    SCCs.push_back(scc);
    omp_unset_lock(&scc_lock);
}

void parTrim(Graph& g) {
    bool changed = true;
    while (changed) {
        changed = false;
        #pragma omp parallel for schedule(dynamic)
        for (int u = 0; u < g.V; ++u) {
            if (mark[u]) continue;
            int in_deg = 0, out_deg = 0;
            for (int i = 0; i < g.V; ++i) {
                for (int j = g.offset[i]; j < g.offset[i+1]; ++j) {
                    if (g.edges[j] == u && !mark[i]) in_deg++;
                    if (i == u && !mark[g.edges[j]]) out_deg++;
                }
            }
            if (in_deg == 0 || out_deg == 0) {
                mark[u] = 1;
                #pragma omp critical
                SCCs.emplace_back(set<int>{u});
                changed = true;
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
        for (int i = 0; i < g.V; ++i) {
            if (mark[i]) continue;
            if (forward) {
                if (u == i) {
                    for (int j = g.offset[i]; j < g.offset[i + 1]; ++j) {
                        int v = g.edges[j];
                        if (!visited[v]) {
                            visited[v] = 1;
                            reachable.push_back(v);
                            q.push(v);
                        }
                    }
                }
            } else {
                for (int j = g.offset[i]; j < g.offset[i + 1]; ++j) {
                    if (g.edges[j] == u && !visited[i]) {
                        visited[i] = 1;
                        reachable.push_back(i);
                        q.push(i);
                    }
                }
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

void parTrim2(Graph& g) {
    #pragma omp parallel for schedule(dynamic)
    for (int u = 0; u < g.V; ++u) {
        if (mark[u]) continue;
        int inNbr = -1, outNbr = -1;
        for (int i = 0; i < g.V; ++i) {
            for (int j = g.offset[i]; j < g.offset[i+1]; ++j) {
                if (g.edges[j] == u && !mark[i]) {
                    if (inNbr == -1) inNbr = i;
                    else inNbr = -2;
                }
            }
        }
        for (int j = g.offset[u]; j < g.offset[u + 1]; ++j) {
            if (!mark[g.edges[j]]) {
                if (outNbr == -1) outNbr = g.edges[j];
                else outNbr = -2;
            }
        }
        if (inNbr >= 0 && inNbr == outNbr) {
            mark[u] = mark[inNbr] = 1;
            #pragma omp critical
            SCCs.emplace_back(set<int>{u, inNbr});
        }
    }
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
}

void method2(Graph& g) {
    color.assign(g.V, 0);
    mark.assign(g.V, 0);
    omp_init_lock(&scc_lock);

    parTrim(g);
    parFWBW(g, 0);
    parTrim2(g);
    parTrim(g);
    parWCC(g);

    #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < g.V; ++i) {
        if (!mark[i])
            recurFWBW(g, i);
    }

    omp_destroy_lock(&scc_lock);
}

vector<pair<int, int>> readSnapEdges(const string& filename, int& maxNode) {
    ifstream in(filename);
    vector<pair<int, int>> edges;
    string line;
    maxNode = 0;
    while (getline(in, line)) {
        if (line[0] == '#') continue;
        istringstream ss(line);
        int u, v;
        ss >> u >> v;
        maxNode = max({maxNode, u, v});
        edges.emplace_back(u, v);
    }
    return edges;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << "p2p-Gnutella04.txt" << endl;
        return 1;
    }
    int maxNode;
    auto edgeList = readSnapEdges(argv[1], maxNode);
    Graph g(maxNode + 1);
    g.buildFromEdges(edgeList);

    method2(g);

    cout << "SCCs found: " << SCCs.size() << endl;
    // for (auto& scc : SCCs) {
    //     for (int v : scc) cout << v << " ";
    //     cout << endl;
    // }
    return 0;
}
