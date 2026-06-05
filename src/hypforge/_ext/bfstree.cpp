// BFS Oblique Tree Engine — C++ backend for HypForge
// Replaces Python BFSObliqueTree to eliminate GPU-CPU sync overhead.
//
// Compile:
//   clang++ -O3 -march=native -shared -fPIC -std=c++17 \
//           src/bfstree_engine.cpp -o models/libbfstree.dylib
//
// Layout conventions (all row-major):
//   Z          : [P, N]           Z[p*N + i]
//   thresholds : [9, P]           thresholds[q*P + p]
//   G, H       : [N, K]           G[i*K + k]
//   leaf_values: [total_nodes, K] leaf_values[t*K + k]

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <climits>
#include <cstdint>

struct BFSTree {
    int max_depth;
    int K;
    int total_nodes;
    std::vector<int>     split_hyp_idx;    // [total_nodes], -1 = leaf
    std::vector<float>   split_threshold;  // [total_nodes]
    std::vector<float>   leaf_values;      // [total_nodes * K]
    std::vector<uint8_t> is_leaf;          // [total_nodes]
};

extern "C" {

// ── Build ────────────────────────────────────────────────────────────────────

void* bfstree_build(
    const float* Z,           // [P, N]
    const float* thresholds,  // [9, P]
    const float* G,           // [N, K]
    const float* H,           // [N, K]
    int P, int N, int K,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    float reg_lambda
) {
    BFSTree* tree     = new BFSTree();
    tree->max_depth   = max_depth;
    tree->K           = K;
    int total_nodes   = (1 << (max_depth + 1)) - 1;
    tree->total_nodes = total_nodes;
    tree->split_hyp_idx.assign(total_nodes, -1);
    tree->split_threshold.assign(total_nodes, 0.0f);
    tree->leaf_values.assign((size_t)total_nodes * K, 0.0f);
    tree->is_leaf.assign(total_nodes, 1);

    // Each node owns its sample index list; routing moves indices to children.
    std::vector<std::vector<int>> node_samples(total_nodes);
    node_samples[0].resize(N);
    std::iota(node_samples[0].begin(), node_samples[0].end(), 0);

    std::vector<float> G_sum(K), H_sum(K);
    std::vector<float> G_tot(K), H_tot(K);
    std::vector<float> G_L(K),   H_L(K);

    for (int depth = 0; depth <= max_depth; depth++) {
        int n_nodes = 1 << depth;
        int base    = n_nodes - 1;
        bool is_last = (depth == max_depth);

        for (int local = 0; local < n_nodes; local++) {
            int t          = base + local;
            auto& samples  = node_samples[t];
            int  n_in      = (int)samples.size();

            if (n_in == 0) {
                tree->is_leaf[t] = 1;
                continue;
            }

            // ── Leaf value (all samples in node) ────────────────────────────
            std::fill(G_sum.begin(), G_sum.end(), 0.0f);
            std::fill(H_sum.begin(), H_sum.end(), 0.0f);
            for (int idx : samples) {
                const float* gi = G + (size_t)idx * K;
                const float* hi = H + (size_t)idx * K;
                for (int k = 0; k < K; k++) { G_sum[k] += gi[k]; H_sum[k] += hi[k]; }
            }
            float* lv = &tree->leaf_values[(size_t)t * K];
            for (int k = 0; k < K; k++)
                lv[k] = -(G_sum[k] / (H_sum[k] + reg_lambda));

            if (is_last || n_in < min_samples_split) {
                tree->is_leaf[t] = 1;
                { std::vector<int> tmp; tmp.swap(samples); }
                continue;
            }

            // ── Subsample for split search ───────────────────────────────────
            const int* sp  = samples.data();
            int        Ns  = n_in;
            std::vector<int> sub_buf;
            if (n_in > 25000) {
                Ns = 25000;
                sub_buf.assign(samples.begin(), samples.begin() + Ns);
                sp = sub_buf.data();
            }

            // G_tot / H_tot on subsample
            std::fill(G_tot.begin(), G_tot.end(), 0.0f);
            std::fill(H_tot.begin(), H_tot.end(), 0.0f);
            for (int j = 0; j < Ns; j++) {
                const float* gi = G + (size_t)sp[j] * K;
                const float* hi = H + (size_t)sp[j] * K;
                for (int k = 0; k < K; k++) { G_tot[k] += gi[k]; H_tot[k] += hi[k]; }
            }

            float parent_score = 0.0f;
            for (int k = 0; k < K; k++)
                parent_score += G_tot[k] * G_tot[k] / (H_tot[k] + reg_lambda);

            // ── Split search: iterate (p, q) ────────────────────────────────
            float best_gain = 1e-5f;
            int   best_p    = -1;
            float best_thr  = 0.0f;

            // Pre-extract z-values for each hypothesis to improve cache hits
            std::vector<float> z_buf(Ns);

            for (int p = 0; p < P; p++) {
                const float* Z_p = Z + (size_t)p * N;
                for (int j = 0; j < Ns; j++) z_buf[j] = Z_p[sp[j]];

                for (int q = 0; q < 9; q++) {
                    float thr = thresholds[q * P + p];

                    std::fill(G_L.begin(), G_L.end(), 0.0f);
                    std::fill(H_L.begin(), H_L.end(), 0.0f);
                    int n_left = 0;

                    for (int j = 0; j < Ns; j++) {
                        if (z_buf[j] < thr) {
                            ++n_left;
                            const float* gi = G + (size_t)sp[j] * K;
                            const float* hi = H + (size_t)sp[j] * K;
                            for (int k = 0; k < K; k++) { G_L[k] += gi[k]; H_L[k] += hi[k]; }
                        }
                    }

                    int n_right = Ns - n_left;
                    if (n_left < min_samples_leaf || n_right < min_samples_leaf) continue;

                    float gain = 0.0f;
                    for (int k = 0; k < K; k++) {
                        float GR = G_tot[k] - G_L[k];
                        float HR = H_tot[k] - H_L[k];
                        gain += G_L[k] * G_L[k] / (H_L[k] + reg_lambda)
                              + GR * GR             / (HR      + reg_lambda);
                    }
                    gain = 0.5f * (gain - parent_score);

                    if (gain > best_gain) {
                        best_gain = gain;
                        best_p    = p;
                        best_thr  = thr;
                    }
                }
            }

            if (best_p < 0) {
                tree->is_leaf[t] = 1;
                { std::vector<int> tmp; tmp.swap(samples); }
                continue;
            }

            // ── Record split & route samples ─────────────────────────────────
            tree->is_leaf[t]         = 0;
            tree->split_hyp_idx[t]   = best_p;
            tree->split_threshold[t] = best_thr;

            int t_left  = 2 * t + 1;
            int t_right = 2 * t + 2;
            const float* Z_best = Z + (size_t)best_p * N;

            node_samples[t_left ].reserve(n_in / 2 + 8);
            node_samples[t_right].reserve(n_in / 2 + 8);
            for (int idx : samples) {
                if (Z_best[idx] < best_thr) node_samples[t_left ].push_back(idx);
                else                        node_samples[t_right].push_back(idx);
            }

            { std::vector<int> tmp; tmp.swap(samples); }
        }
    }

    return static_cast<void*>(tree);
}

// ── Predict ──────────────────────────────────────────────────────────────────

void bfstree_predict(
    void*        handle,
    const float* Z_pred,  // [P, N_pred]
    int          N_pred,
    float*       out      // [N_pred, K]  (caller-allocated, zero-initialised)
) {
    const BFSTree* tree = static_cast<const BFSTree*>(handle);
    int K         = tree->K;
    int max_depth = tree->max_depth;

    std::vector<int> node_assign(N_pred, 0);

    for (int depth = 0; depth < max_depth; depth++) {
        int n_nodes = 1 << depth;
        int base    = n_nodes - 1;

        for (int local = 0; local < n_nodes; local++) {
            int t = base + local;
            if (tree->is_leaf[t]) continue;

            int         p   = tree->split_hyp_idx[t];
            float       thr = tree->split_threshold[t];
            const float* Zp = Z_pred + (size_t)p * N_pred;
            int t_left  = 2 * t + 1;
            int t_right = 2 * t + 2;

            for (int i = 0; i < N_pred; i++) {
                if (node_assign[i] == t)
                    node_assign[i] = (Zp[i] < thr) ? t_left : t_right;
            }
        }
    }

    for (int i = 0; i < N_pred; i++) {
        const float* lv = &tree->leaf_values[(size_t)node_assign[i] * K];
        float*       oi = out + (size_t)i * K;
        for (int k = 0; k < K; k++) oi[k] = lv[k];
    }
}

// ── Lifecycle ────────────────────────────────────────────────────────────────

void bfstree_free(void* handle) {
    delete static_cast<BFSTree*>(handle);
}

int bfstree_get_K(void* handle) {
    return static_cast<BFSTree*>(handle)->K;
}
int bfstree_get_max_depth(void* handle) {
    return static_cast<BFSTree*>(handle)->max_depth;
}
int bfstree_get_total_nodes(void* handle) {
    return static_cast<BFSTree*>(handle)->total_nodes;
}

// Fill out[i] with split_hyp_idx[i] for i in [0, total_nodes).
// out must be caller-allocated int array of size total_nodes.
void bfstree_get_split_indices(void* handle, int* out) {
    const BFSTree* tree = static_cast<const BFSTree*>(handle);
    for (int i = 0; i < tree->total_nodes; ++i)
        out[i] = tree->split_hyp_idx[i];
}

} // extern "C"
