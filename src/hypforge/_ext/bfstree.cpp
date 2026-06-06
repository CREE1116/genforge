// BFS Oblique Tree Engine — C++ backend for HypForge
// Replaces Python BFSObliqueTree to eliminate GPU-CPU sync overhead.
//
// Compile (with OpenMP on Mac):
//   OMP=$(brew --prefix libomp)
//   clang++ -O3 -march=native -shared -fPIC -std=c++17 \
//           -Xpreprocessor -fopenmp -I$OMP/include -L$OMP/lib -lomp \
//           bfstree.cpp -o libbfstree.dylib
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
#include <random>
#include <unordered_set>
#ifdef _OPENMP
#  include <omp.h>
#endif

struct Hypothesis {
    int type; // 0=linear, 1=square, 2=abs, 3=product
    std::vector<float> w; // size D
    int h1_idx = -1;
    int h2_idx = -1;

    // bandit stats
    int n_obs = 0;
    double mu_fitness = 0.0;
    double M2_fitness = 0.0;

    // usage stats
    int use_count = 0;
    int rounds_since_last_use = 0;

    float fitness = 0.0;
    float score = 0.0;
    bool is_base = false;

    // caches
    std::vector<float> full_cache; // size N
    std::vector<float> thresholds; // size 9
    int stored_complexity = 0;

    int complexity() const {
        return stored_complexity;
    }

    void update_complexity(const std::vector<Hypothesis>& pool) {
        if (type == 3) {
            stored_complexity = pool[h1_idx].complexity() + pool[h2_idx].complexity();
        } else {
            stored_complexity = 0;
            for (float val : w) {
                if (std::abs(val) > 1e-3f) stored_complexity++;
            }
        }
    }

    float ucb_score(float eta_penalty) const {
        if (n_obs == 0) return 0.0f;
        float n = (float)n_obs;
        float sigma = (n_obs > 1) ? std::sqrt((float)(M2_fitness / n)) : 0.0f;
        float bonus = (1.0f * sigma + 0.5f) / std::sqrt(n);
        return (float)mu_fitness + bonus - eta_penalty * stored_complexity;
    }

    void observe_fitness(float fit_val, float eta_penalty) {
        fitness = fit_val;
        n_obs++;
        double delta = fit_val - mu_fitness;
        mu_fitness += delta / n_obs;
        double delta2 = fit_val - mu_fitness;
        M2_fitness += delta * delta2;
        score = ucb_score(eta_penalty);
    }
};

static std::vector<int> get_random_indices(int N, int size, unsigned int seed) {
    std::vector<int> idxs(N);
    std::iota(idxs.begin(), idxs.end(), 0);
    if (N <= size) return idxs;
    std::mt19937 g(seed);
    for (int i = 0; i < size; i++) {
        std::uniform_int_distribution<int> dist(i, N - 1);
        int j = dist(g);
        std::swap(idxs[i], idxs[j]);
    }
    idxs.resize(size);
    return idxs;
}

class HypForgePool {
public:
    int D;
    int max_size;
    int N = 0;
    std::vector<Hypothesis> pop;
    unsigned int seed = 42;

    HypForgePool(int D, int max_size) : D(D), max_size(max_size) {
        // Initialize base hypotheses
        for (int j = 0; j < D; j++) {
            Hypothesis h;
            h.type = 0; // linear
            h.w.assign(D, 0.0f);
            h.w[j] = 1.0f;
            h.is_base = true;
            h.update_complexity(pop);
            pop.push_back(h);
        }
    }

    void initialize_cache(Hypothesis& h, const float* X, const std::vector<int>& rand_indices) {
        h.full_cache.resize(N);
        if (h.type == 3) {
            for (int i = 0; i < N; i++) {
                h.full_cache[i] = pop[h.h1_idx].full_cache[i] * pop[h.h2_idx].full_cache[i];
            }
        } else {
            for (int i = 0; i < N; i++) {
                float val = 0.0f;
                for (int d = 0; d < D; d++) {
                    val += h.w[d] * X[i * D + d];
                }
                if (h.type == 0) h.full_cache[i] = val;
                else if (h.type == 1) h.full_cache[i] = val * val;
                else if (h.type == 2) h.full_cache[i] = std::abs(val);
            }
        }

        int Ns_q = rand_indices.size();
        std::vector<float> sample(Ns_q);
        for (int i = 0; i < Ns_q; i++) {
            sample[i] = h.full_cache[rand_indices[i]];
        }
        std::sort(sample.begin(), sample.end());

        h.thresholds.resize(9);
        for (int q = 0; q < 9; q++) {
            int idx = (int)((q + 1) * 0.1f * Ns_q);
            if (idx >= Ns_q) idx = Ns_q - 1;
            h.thresholds[q] = sample[idx];
        }
    }

    std::vector<float> sparsify(const std::vector<float>& w, int D_num) {
        int D_in = w.size();
        std::vector<float> w_sp = w;
        if (D_num < D_in) {
            for (int i = D_num; i < D_in; i++) w_sp[i] = 0.0f;
        }
        int k = std::max(2, (int)std::sqrt(D_num));
        if (D_num > k) {
            std::vector<std::pair<float, int>> abs_vals(D_num);
            for (int i = 0; i < D_num; i++) {
                abs_vals[i] = {std::abs(w_sp[i]), i};
            }
            std::sort(abs_vals.begin(), abs_vals.end(), [](const auto& a, const auto& b) {
                return a.first > b.first;
            });
            std::vector<bool> keep(D_num, false);
            for (int i = 0; i < k; i++) {
                keep[abs_vals[i].second] = true;
            }
            for (int i = 0; i < D_num; i++) {
                if (!keep[i]) w_sp[i] = 0.0f;
            }
        }

        float norm = 0.0f;
        for (float val : w_sp) norm += val * val;
        norm = std::sqrt(norm);
        if (norm > 1e-5f) {
            for (float& val : w_sp) val /= norm;
            return w_sp;
        }

        std::vector<float> fallback(D_in, 0.0f);
        std::mt19937 g(seed++);
        fallback[g() % D_num] = 1.0f;
        return fallback;
    }

    void evolve(
        const float* X,
        const float* G_full,
        const float* H_full,
        const int* sub_indices,
        int N_in, int Ns, int K,
        int D_num,
        float reg_lambda,
        float eta_penalty = 0.002f
    ) {
        this->N = N_in;
        std::vector<int> rand_indices = get_random_indices(N, std::min(N, 10000), seed++);
        for (auto& h : pop) {
            if (h.full_cache.empty()) {
                initialize_cache(h, X, rand_indices);
            }
        }

        std::vector<float> G_tot(K, 0.0f), H_tot(K, 0.0f);
        for (int j = 0; j < Ns; j++) {
            int idx = sub_indices[j];
            for (int k = 0; k < K; k++) {
                G_tot[k] += G_full[idx * K + k];
                H_tot[k] += H_full[idx * K + k];
            }
        }
        float parent_score = 0.0f;
        for (int k = 0; k < K; k++) {
            parent_score += G_tot[k] * G_tot[k] / (H_tot[k] + reg_lambda);
        }

        int P = pop.size();
        for (int p = 0; p < P; p++) {
            auto& h = pop[p];
            float best_gain = 0.0f;
            for (int q = 0; q < 9; q++) {
                float thr = h.thresholds[q];
                std::vector<float> G_L(K, 0.0f), H_L(K, 0.0f);
                int n_left = 0;
                for (int j = 0; j < Ns; j++) {
                    int idx = sub_indices[j];
                    if (h.full_cache[idx] < thr) {
                        n_left++;
                        for (int k = 0; k < K; k++) {
                            G_L[k] += G_full[idx * K + k];
                            H_L[k] += H_full[idx * K + k];
                        }
                    }
                }
                int n_right = Ns - n_left;
                if (n_left < 10 || n_right < 10) continue;
                float gain = 0.0f;
                for (int k = 0; k < K; k++) {
                    float GR = G_tot[k] - G_L[k];
                    float HR = H_tot[k] - H_L[k];
                    gain += G_L[k] * G_L[k] / (H_L[k] + reg_lambda)
                          + GR * GR / (HR + reg_lambda);
                }
                gain = 0.5f * (gain - parent_score);
                if (gain > best_gain) best_gain = gain;
            }
            h.observe_fitness(best_gain, eta_penalty);
        }

        std::sort(pop.begin(), pop.end(), [](const Hypothesis& a, const Hypothesis& b) {
            return a.score > b.score;
        });

        std::vector<Hypothesis> raw_candidates;

        std::vector<float> V(D * K, 0.0f);
        for (int d = 0; d < D; d++) {
            for (int k = 0; k < K; k++) {
                float sum_val = 0.0f;
                for (int j = 0; j < Ns; j++) {
                    int idx = sub_indices[j];
                    sum_val += X[idx * D + d] * G_full[idx * K + k];
                }
                V[d * K + k] = sum_val;
            }
        }

        for (int k = 0; k < K; k++) {
            std::vector<float> V_k(D);
            for (int d = 0; d < D; d++) V_k[d] = V[d * K + k];
            std::vector<float> w_k = sparsify(V_k, D_num);
            for (int type = 0; type < 3; type++) {
                Hypothesis c;
                c.type = type;
                c.w = w_k;
                c.update_complexity(pop);
                raw_candidates.push_back(c);
            }
        }

        int flow_count = 0;
        for (int i = 0; i < (int)pop.size(); i++) {
            if (flow_count >= 20) break;
            const auto& h = pop[i];
            if (h.type >= 3) continue;
            flow_count++;

            std::vector<float> z(Ns);
            std::vector<float> p_proj(Ns);
            for (int j = 0; j < Ns; j++) {
                int idx = sub_indices[j];
                float val = 0.0f;
                for (int d = 0; d < D; d++) val += h.w[d] * X[idx * D + d];
                p_proj[j] = val;
                if (h.type == 0) z[j] = val;
                else if (h.type == 1) z[j] = val * val;
                else if (h.type == 2) z[j] = std::abs(val);
            }

            std::vector<float> v(K, 0.0f);
            for (int k = 0; k < K; k++) {
                for (int j = 0; j < Ns; j++) {
                    int idx = sub_indices[j];
                    v[k] += G_full[idx * K + k] * z[j];
                }
            }

            std::vector<float> g(Ns, 0.0f);
            for (int j = 0; j < Ns; j++) {
                int idx = sub_indices[j];
                for (int k = 0; k < K; k++) {
                    g[j] += G_full[idx * K + k] * v[k];
                }
            }

            std::vector<float> grad_w(D, 0.0f);
            for (int d = 0; d < D; d++) {
                float sum_grad = 0.0f;
                for (int j = 0; j < Ns; j++) {
                    int idx = sub_indices[j];
                    float diff = g[j] - z[j];
                    if (h.type == 0) {
                        sum_grad += X[idx * D + d] * diff;
                    } else if (h.type == 1) {
                        sum_grad += 2.0f * X[idx * D + d] * diff * p_proj[j];
                    } else if (h.type == 2) {
                        float sign_p = (p_proj[j] >= 0.0f) ? 1.0f : -1.0f;
                        sum_grad += X[idx * D + d] * diff * sign_p;
                    }
                }
                grad_w[d] = sum_grad;
            }

            std::vector<float> w_new(D);
            for (int d = 0; d < D; d++) w_new[d] = h.w[d] + 0.1f * grad_w[d];
            std::vector<float> w_refined = sparsify(w_new, D_num);
            Hypothesis c;
            c.type = h.type;
            c.w = w_refined;
            c.update_complexity(pop);
            raw_candidates.push_back(c);
        }

        if (raw_candidates.empty()) return;

        std::vector<Hypothesis> admitted;
        int C_cand = raw_candidates.size();

        std::vector<std::vector<float>> cand_sub_cache(C_cand, std::vector<float>(Ns));
        for (int ci = 0; ci < C_cand; ci++) {
            auto& c = raw_candidates[ci];
            for (int j = 0; j < Ns; j++) {
                int idx = sub_indices[j];
                float val = 0.0f;
                for (int d = 0; d < D; d++) val += c.w[d] * X[idx * D + d];
                if (c.type == 0) cand_sub_cache[ci][j] = val;
                else if (c.type == 1) cand_sub_cache[ci][j] = val * val;
                else if (c.type == 2) cand_sub_cache[ci][j] = std::abs(val);
            }
        }

        for (int ci = 0; ci < C_cand; ci++) {
            auto& c = raw_candidates[ci];
            const auto& z_sub = cand_sub_cache[ci];

            std::vector<float> sorted_z = z_sub;
            std::sort(sorted_z.begin(), sorted_z.end());
            std::vector<float> cand_thr(9);
            for (int q = 0; q < 9; q++) {
                int idx = (int)((q + 1) * 0.1f * Ns);
                if (idx >= Ns) idx = Ns - 1;
                cand_thr[q] = sorted_z[idx];
            }

            float best_gain = 0.0f;
            for (int q = 0; q < 9; q++) {
                float thr = cand_thr[q];
                std::vector<float> G_L(K, 0.0f), H_L(K, 0.0f);
                int n_left = 0;
                for (int j = 0; j < Ns; j++) {
                    if (z_sub[j] < thr) {
                        n_left++;
                        int idx = sub_indices[j];
                        for (int k = 0; k < K; k++) {
                            G_L[k] += G_full[idx * K + k];
                            H_L[k] += H_full[idx * K + k];
                        }
                    }
                }
                int n_right = Ns - n_left;
                if (n_left < 10 || n_right < 10) continue;
                float gain = 0.0f;
                for (int k = 0; k < K; k++) {
                    float GR = G_tot[k] - G_L[k];
                    float HR = H_tot[k] - H_L[k];
                    gain += G_L[k] * G_L[k] / (H_L[k] + reg_lambda)
                          + GR * GR / (HR + reg_lambda);
                }
                gain = 0.5f * (gain - parent_score);
                if (gain > best_gain) best_gain = gain;
            }

            if (best_gain > 1e-5f) {
                c.observe_fitness(best_gain, eta_penalty);
                admitted.push_back(c);
            }
        }

        for (auto& c : admitted) {
            initialize_cache(c, X, rand_indices);
            pop.push_back(c);
        }

        for (auto& h : pop) {
            h.score = h.ucb_score(eta_penalty);
        }
        std::sort(pop.begin(), pop.end(), [](const Hypothesis& a, const Hypothesis& b) {
            return a.score > b.score;
        });

        P = pop.size();
        std::vector<std::vector<float>> sub_caches(P, std::vector<float>(Ns));
        std::vector<float> norms(P, 0.0f);
        for (int p = 0; p < P; p++) {
            float norm_sq = 0.0f;
            for (int j = 0; j < Ns; j++) {
                int idx = sub_indices[j];
                float val = pop[p].full_cache[idx];
                sub_caches[p][j] = val;
                norm_sq += val * val;
            }
            norms[p] = std::sqrt(norm_sq) + 1e-8f;
        }

        std::vector<int> kept;
        for (int i = 0; i < P; i++) {
            bool redundant = false;
            for (int j : kept) {
                float thr_val = (pop[i].type == pop[j].type) ? 0.90f : 0.98f;
                float dot = 0.0f;
                for (int k = 0; k < Ns; k++) {
                    dot += sub_caches[i][k] * sub_caches[j][k];
                }
                float similarity = std::abs(dot) / (norms[i] * norms[j]);
                if (similarity > thr_val) {
                    redundant = true;
                    break;
                }
            }
            if (!redundant) {
                kept.push_back(i);
                if ((int)kept.size() >= max_size) break;
            }
        }

        std::vector<Hypothesis> pruned_pop;
        for (int i : kept) {
            pruned_pop.push_back(std::move(pop[i]));
        }
        pop = std::move(pruned_pop);

        std::vector<Hypothesis> survivor_pop;
        for (auto& h : pop) {
            if (h.is_base || h.n_obs < 3 || h.mu_fitness + h.score > 1e-6f) {
                survivor_pop.push_back(std::move(h));
            }
        }
        pop = std::move(survivor_pop);
    }

    void eval(const float* X, int N_in, float* out_Z) const {
        int P_size = pop.size();
        for (int i = 0; i < N_in; i++) {
            for (int p = 0; p < P_size; p++) {
                const auto& h = pop[p];
                if (h.type == 3) {
                    out_Z[(size_t)p * N_in + i] = out_Z[(size_t)h.h1_idx * N_in + i] * out_Z[(size_t)h.h2_idx * N_in + i];
                } else {
                    float val = 0.0f;
                    for (int d = 0; d < D; d++) {
                        val += h.w[d] * X[i * D + d];
                    }
                    if (h.type == 0) out_Z[(size_t)p * N_in + i] = val;
                    else if (h.type == 1) out_Z[(size_t)p * N_in + i] = val * val;
                    else if (h.type == 2) out_Z[(size_t)p * N_in + i] = std::abs(val);
                }
            }
        }
    }
};

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

            // ── Split search: parallel over hypotheses ───────────────────────
            float best_gain = 1e-5f;
            int   best_p    = -1;
            float best_thr  = 0.0f;

#ifdef _OPENMP
            #pragma omp parallel
            {
                // Thread-local buffers — avoids false sharing
                std::vector<float> z_buf_tl(Ns);
                std::vector<float> G_L_tl(K, 0.0f), H_L_tl(K, 0.0f);
                float tl_gain = 1e-5f;
                int   tl_p    = -1;
                float tl_thr  = 0.0f;

                #pragma omp for nowait
                for (int p = 0; p < P; p++) {
                    const float* Z_p = Z + (size_t)p * N;
                    for (int j = 0; j < Ns; j++) z_buf_tl[j] = Z_p[sp[j]];

                    for (int q = 0; q < 9; q++) {
                        float thr = thresholds[q * P + p];
                        std::fill(G_L_tl.begin(), G_L_tl.end(), 0.0f);
                        std::fill(H_L_tl.begin(), H_L_tl.end(), 0.0f);
                        int n_left = 0;

                        for (int j = 0; j < Ns; j++) {
                            if (z_buf_tl[j] < thr) {
                                ++n_left;
                                const float* gi = G + (size_t)sp[j] * K;
                                const float* hi = H + (size_t)sp[j] * K;
                                for (int k = 0; k < K; k++) { G_L_tl[k] += gi[k]; H_L_tl[k] += hi[k]; }
                            }
                        }

                        int n_right = Ns - n_left;
                        if (n_left < min_samples_leaf || n_right < min_samples_leaf) continue;

                        float gain = 0.0f;
                        for (int k = 0; k < K; k++) {
                            float GR = G_tot[k] - G_L_tl[k];
                            float HR = H_tot[k] - H_L_tl[k];
                            gain += G_L_tl[k] * G_L_tl[k] / (H_L_tl[k] + reg_lambda)
                                  + GR * GR               / (HR         + reg_lambda);
                        }
                        gain = 0.5f * (gain - parent_score);

                        if (gain > tl_gain) { tl_gain = gain; tl_p = p; tl_thr = thr; }
                    }
                }

                #pragma omp critical
                {
                    if (tl_gain > best_gain) { best_gain = tl_gain; best_p = tl_p; best_thr = tl_thr; }
                }
            }
#else
            // Fallback: serial (no OpenMP)
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
                              + GR * GR           / (HR      + reg_lambda);
                    }
                    gain = 0.5f * (gain - parent_score);

                    if (gain > best_gain) { best_gain = gain; best_p = p; best_thr = thr; }
                }
            }
#endif

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

// ── Serialization ─────────────────────────────────────────────────────────────

// Export all tree arrays for serialization.
// Caller allocates: split_hyp_idx[total_nodes], split_threshold[total_nodes],
//                   leaf_values[total_nodes * K], is_leaf[total_nodes].
void bfstree_export(
    void*    handle,
    int*     split_hyp_idx,
    float*   split_threshold,
    float*   leaf_values,
    uint8_t* is_leaf
) {
    const BFSTree* tree = static_cast<const BFSTree*>(handle);
    int n   = tree->total_nodes;
    int K   = tree->K;
    for (int i = 0; i < n;   ++i) split_hyp_idx[i]  = tree->split_hyp_idx[i];
    for (int i = 0; i < n;   ++i) split_threshold[i] = tree->split_threshold[i];
    for (int i = 0; i < n*K; ++i) leaf_values[i]     = tree->leaf_values[i];
    for (int i = 0; i < n;   ++i) is_leaf[i]          = tree->is_leaf[i];
}

// Reconstruct a tree from serialized arrays (no training).
void* bfstree_from_arrays(
    const int*     split_hyp_idx,
    const float*   split_threshold,
    const float*   leaf_values,
    const uint8_t* is_leaf,
    int total_nodes, int K, int max_depth
) {
    BFSTree* tree     = new BFSTree();
    tree->total_nodes = total_nodes;
    tree->K           = K;
    tree->max_depth   = max_depth;
    tree->split_hyp_idx.assign(split_hyp_idx,   split_hyp_idx   + total_nodes);
    tree->split_threshold.assign(split_threshold, split_threshold + total_nodes);
    tree->leaf_values.assign(leaf_values,        leaf_values      + (size_t)total_nodes * K);
    tree->is_leaf.assign(is_leaf,               is_leaf           + total_nodes);
    return static_cast<void*>(tree);
}

// ── Pool Lifecycle ───────────────────────────────────────────────────────────

void* pool_create(int D, int max_size) {
    return static_cast<void*>(new HypForgePool(D, max_size));
}

void pool_free(void* handle) {
    delete static_cast<HypForgePool*>(handle);
}

void pool_evolve(
    void* handle,
    const float* X,
    const float* G_full,
    const float* H_full,
    const int* sub_indices,
    int N_in, int Ns, int K,
    int D_num,
    float reg_lambda,
    float eta_penalty
) {
    static_cast<HypForgePool*>(handle)->evolve(
        X, G_full, H_full, sub_indices, N_in, Ns, K, D_num, reg_lambda, eta_penalty
    );
}

void pool_eval(void* handle, const float* X, int N_in, float* out_Z) {
    static_cast<const HypForgePool*>(handle)->eval(X, N_in, out_Z);
}

int pool_get_size(void* handle) {
    return static_cast<const HypForgePool*>(handle)->pop.size();
}

void pool_get_caches_and_thresholds(void* handle, float* out_Z, float* out_thresholds) {
    auto* pool = static_cast<HypForgePool*>(handle);
    int P = pool->pop.size();
    int N = pool->N;
    for (int p = 0; p < P; p++) {
        std::memcpy(out_Z + (size_t)p * N, pool->pop[p].full_cache.data(), N * sizeof(float));
        for (int q = 0; q < 9; q++) {
            out_thresholds[q * P + p] = pool->pop[p].thresholds[q];
        }
    }
}

void pool_update_use_counts(void* handle, const int* split_indices, int n_nodes) {
    auto* pool = static_cast<HypForgePool*>(handle);
    int P = pool->pop.size();
    for (int p = 0; p < P; p++) {
        pool->pop[p].rounds_since_last_use++;
    }
    for (int i = 0; i < n_nodes; i++) {
        int idx = split_indices[i];
        if (idx >= 0 && idx < P) {
            pool->pop[idx].use_count++;
            pool->pop[idx].rounds_since_last_use = 0;
        }
    }
}

void pool_export(
    void* handle,
    int* types,
    float* weights,
    int* h1_indices,
    int* h2_indices,
    int* n_obs,
    double* mu_fitness,
    double* M2_fitness,
    int* use_counts,
    int* rounds_since_last_use,
    float* fitnesses,
    float* scores,
    uint8_t* is_base
) {
    auto* pool = static_cast<HypForgePool*>(handle);
    int P = pool->pop.size();
    int D = pool->D;
    for (int p = 0; p < P; p++) {
        const auto& h = pool->pop[p];
        types[p] = h.type;
        h1_indices[p] = h.h1_idx;
        h2_indices[p] = h.h2_idx;
        n_obs[p] = h.n_obs;
        mu_fitness[p] = h.mu_fitness;
        M2_fitness[p] = h.M2_fitness;
        use_counts[p] = h.use_count;
        rounds_since_last_use[p] = h.rounds_since_last_use;
        fitnesses[p] = h.fitness;
        scores[p] = h.score;
        is_base[p] = h.is_base ? 1 : 0;
        
        if (h.type < 3) {
            std::memcpy(weights + p * D, h.w.data(), D * sizeof(float));
        } else {
            std::memset(weights + p * D, 0, D * sizeof(float));
        }
    }
}

void* pool_import(
    int D, int max_size, int P,
    const int* types,
    const float* weights,
    const int* h1_indices,
    const int* h2_indices,
    const int* n_obs,
    const double* mu_fitness,
    const double* M2_fitness,
    const int* use_counts,
    const int* rounds_since_last_use,
    const float* fitnesses,
    const float* scores,
    const uint8_t* is_base
) {
    auto* pool = new HypForgePool(D, max_size);
    pool->pop.clear();
    for (int p = 0; p < P; p++) {
        Hypothesis h;
        h.type = types[p];
        h.h1_idx = h1_indices[p];
        h.h2_idx = h2_indices[p];
        h.n_obs = n_obs[p];
        h.mu_fitness = mu_fitness[p];
        h.M2_fitness = M2_fitness[p];
        h.use_count = use_counts[p];
        h.rounds_since_last_use = rounds_since_last_use[p];
        h.fitness = fitnesses[p];
        h.score = scores[p];
        h.is_base = is_base[p] != 0;
        
        if (h.type < 3) {
            h.w.assign(weights + p * D, weights + (p + 1) * D);
        }
        h.update_complexity(pool->pop);
        pool->pop.push_back(std::move(h));
    }
    return static_cast<void*>(pool);
}

} // extern "C"

