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

// Operator types:
//   0 = linear     : w·x
//   1 = leaky_relu : max(w·x, 0.01·w·x)
//   2 = product    : h1·h2  (composite)
static inline float apply_op(float val, int type) {
    if (type == 1) return (val > 0.0f) ? val : 0.01f * val;
    return val; // 0=linear, default
}

struct Hypothesis {
    int type; // 0=linear, 1=leaky_relu (alpha=0.01), 2=product
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
    std::vector<float> thresholds; // size 9 (kept for backward-compat / evolve fitness scan)
    float bin_min = 0.0f;          // uniform-histogram range, computed in initialize_cache
    float bin_max = 1.0f;
    int stored_complexity = 0;

    int complexity() const {
        return stored_complexity;
    }

    void update_complexity(const std::vector<Hypothesis>& pool) {
        if (type == 2) {
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

// evolve_mode      : 0=standard, 1=crossover, 2=novelty
// op_mode          : 0=all (linear+lrelu), 1=linear_only, 2=lrelu_only
// mutation_mode    : 0=gradient, 1=random_perturb, 2=none
// feedback_mode    : 0=split_gain, 1=scan_only
// crossover_top_k  : top-k pairs for crossover
// fitness_norm_mode: 0=none, 1=round_mean, 2=zscore
// elitism_k        : always protect top-k from similarity eviction
// alps_mode        : 0=off, 1=youth-priority (n_obs<5 get score boost during eviction)

class HypForgePool {
public:
    int D;
    int max_size;
    int N = 0;
    int op_mode           = 0;
    int crossover_top_k   = 6;
    int elitism_k            = 0;
    int alps_mode            = 0;
    int map_elites_slots     = 0;   // >0: max per type in pool
    std::vector<Hypothesis> pop;
    unsigned int seed = 42;

    // Persistent scratch buffers — allocated once, reused every round.
    // Avoids the cost of repeated 30-50 MB heap allocations per round.
    std::vector<float> Z_buf_;   // [P * Ns_tree] subsampled projections
    std::vector<float> G_buf_;   // [Ns_tree * K]
    std::vector<float> H_buf_;   // [Ns_tree * K]

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
        if (h.type == 2) {
            for (int i = 0; i < N; i++) {
                h.full_cache[i] = pop[h.h1_idx].full_cache[i] * pop[h.h2_idx].full_cache[i];
            }
        } else {
            for (int i = 0; i < N; i++) {
                float val = 0.0f;
                for (int d = 0; d < D; d++) val += h.w[d] * X[i * D + d];
                h.full_cache[i] = apply_op(val, h.type);
            }
        }

        int Ns_q = rand_indices.size();
        std::vector<float> sample(Ns_q);
        for (int i = 0; i < Ns_q; i++) {
            sample[i] = h.full_cache[rand_indices[i]];
        }
        std::sort(sample.begin(), sample.end());

        // 9 quantile thresholds for the evolve fitness scan (kept for backward compat)
        h.thresholds.resize(9);
        for (int q = 0; q < 9; q++) {
            int idx = (int)((q + 1) * 0.1f * Ns_q);
            if (idx >= Ns_q) idx = Ns_q - 1;
            h.thresholds[q] = sample[idx];
        }

        // min/max range for uniform histogram binning in pool_build_tree
        h.bin_min = sample.front();
        h.bin_max = sample.back();
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
        std::vector<float> best_gains(P, 0.0f);
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 2)
#endif
        for (int p = 0; p < P; p++) {
            const auto& h = pop[p];
            float best_gain = 0.0f;
            std::vector<float> G_L(K), H_L(K);
            for (int q = 0; q < 9; q++) {
                float thr = h.thresholds[q];
                std::fill(G_L.begin(), G_L.end(), 0.0f);
                std::fill(H_L.begin(), H_L.end(), 0.0f);
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
            best_gains[p] = best_gain;
        }

        for (int p = 0; p < P; p++) {
            pop[p].observe_fitness(best_gains[p], eta_penalty);
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
            // op_mode: 0=linear+lrelu, 1=linear_only, 2=lrelu_only
            int type_lo = (op_mode == 2) ? 1 : 0;
            int type_hi = (op_mode == 1) ? 1 : 2;
            for (int type = type_lo; type < type_hi; type++) {
                Hypothesis c;
                c.type = type;
                c.w = w_k;
                c.update_complexity(pop);
                raw_candidates.push_back(c);
            }
        }

        // Crossover between top hypothesis pairs
        if ((int)pop.size() >= 2) {
            std::mt19937 rng_cx(seed++);
            std::uniform_real_distribution<float> alpha_dist(0.3f, 0.7f);
            int n_top = std::min((int)pop.size(), crossover_top_k);
            for (int i = 0; i < n_top; i++) {
                for (int j = i + 1; j < n_top; j++) {
                    if (pop[i].type >= 2 || pop[j].type >= 2) continue; // skip product
                    float alpha = alpha_dist(rng_cx);
                    std::vector<float> w_child(D);
                    for (int d = 0; d < D; d++)
                        w_child[d] = alpha * pop[i].w[d] + (1.0f - alpha) * pop[j].w[d];
                    w_child = sparsify(w_child, D_num);
                    // child inherits the type of the fitter parent
                    Hypothesis c;
                    c.type = (pop[i].score >= pop[j].score) ? pop[i].type : pop[j].type;
                    c.w = w_child;
                    c.update_complexity(pop);
                    raw_candidates.push_back(c);
                }
            }
        }

        if (raw_candidates.empty()) return;

        std::vector<Hypothesis> admitted;
        int C_cand = raw_candidates.size();

        std::vector<std::vector<float>> cand_sub_cache(C_cand, std::vector<float>(Ns));
        std::vector<float> cand_best_gain(C_cand, 0.0f);

        // Parallelize both sub-cache computation and fitness scan over candidates.
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 2)
#endif
        for (int ci = 0; ci < C_cand; ci++) {
            const auto& c = raw_candidates[ci];
            auto& z_sub = cand_sub_cache[ci];
            for (int j = 0; j < Ns; j++) {
                int idx = sub_indices[j];
                float val = 0.0f;
                for (int d = 0; d < D; d++) val += c.w[d] * X[idx * D + d];
                z_sub[j] = apply_op(val, c.type);
            }

            std::vector<float> sorted_z = z_sub;
            std::sort(sorted_z.begin(), sorted_z.end());
            std::vector<float> cand_thr(9);
            for (int q = 0; q < 9; q++) {
                int idx2 = (int)((q + 1) * 0.1f * Ns);
                if (idx2 >= Ns) idx2 = Ns - 1;
                cand_thr[q] = sorted_z[idx2];
            }

            float best_gain = 0.0f;
            std::vector<float> G_L(K), H_L(K);
            for (int q = 0; q < 9; q++) {
                float thr = cand_thr[q];
                std::fill(G_L.begin(), G_L.end(), 0.0f);
                std::fill(H_L.begin(), H_L.end(), 0.0f);
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
            cand_best_gain[ci] = best_gain;
        }

        for (int ci = 0; ci < C_cand; ci++) {
            if (cand_best_gain[ci] > 1e-5f) {
                auto& c = raw_candidates[ci];
                c.observe_fitness(cand_best_gain[ci], eta_penalty);
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
        // Extract and normalize sub_caches on a smaller subset of samples
        int Ns_sim = std::min(Ns, 1000);
        std::vector<std::vector<float>> sub_caches(P, std::vector<float>(Ns_sim));
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int p = 0; p < P; p++) {
            float norm_sq = 0.0f;
            for (int j = 0; j < Ns_sim; j++) {
                int idx = sub_indices[j];
                float val = pop[p].full_cache[idx];
                sub_caches[p][j] = val;
                norm_sq += val * val;
            }
            float norm = std::sqrt(norm_sq) + 1e-8f;
            for (int j = 0; j < Ns_sim; j++) {
                sub_caches[p][j] /= norm;
            }
        }

        // Precompute the absolute similarity matrix in parallel
        std::vector<float> sim_matrix(P * P, 0.0f);
#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 8)
#endif
        for (int i = 0; i < P; i++) {
            for (int j = i + 1; j < P; j++) {
                float dot = 0.0f;
                const float* ci = sub_caches[i].data();
                const float* cj = sub_caches[j].data();
                for (int k = 0; k < Ns_sim; k++) {
                    dot += ci[k] * cj[k];
                }
                float sim = std::abs(dot);
                sim_matrix[i * P + j] = sim;
                sim_matrix[j * P + i] = sim;
            }
        }

        // Priority boosts for eviction ordering: ALPS (youth)
        if (alps_mode > 0) {
            for (auto& h : pop)
                if (h.n_obs < 5) h.score += 1e6f;
            std::sort(pop.begin(), pop.end(), [](const Hypothesis& a, const Hypothesis& b) {
                return a.score > b.score;
            });
            for (auto& h : pop)
                if (h.n_obs < 5) h.score -= 1e6f;
        }

        std::vector<int> kept;
        int elite_cap = (elitism_k > 0) ? std::min(elitism_k, P) : 0;
        for (int i = 0; i < elite_cap; i++) kept.push_back(i);

        for (int i = 0; i < P; i++) {
            if (i < elite_cap) continue;
            bool redundant = false;
            for (int j : kept) {
                float thr_val = (pop[i].type == pop[j].type) ? 0.90f : 0.98f;
                if (sim_matrix[i * P + j] > thr_val) {
                    redundant = true;
                    break;
                }
            }
            if (!redundant) {
                kept.push_back(i);
                if ((int)kept.size() >= max_size) break;
            }
        }

        // MAP-Elites: apply per-type slot quota after similarity dedup
        if (map_elites_slots > 0) {
            std::vector<int> filtered;
            int type_cnts[3] = {0, 0, 0};
            for (int i : kept) {
                int t = pop[i].type;
                if (type_cnts[t < 3 ? t : 0] < map_elites_slots) {
                    filtered.push_back(i);
                    if (t >= 0 && t < 3) type_cnts[t]++;
                }
            }
            kept = filtered;
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
        // Parallelise over samples; inner p-loop is sequential because type==2 (product) reads p' < p.
#ifdef _OPENMP
        #pragma omp parallel for schedule(static, 256)
#endif
        for (int i = 0; i < N_in; i++) {
            for (int p = 0; p < P_size; p++) {
                const auto& h = pop[p];
                if (h.type == 2) {
                    out_Z[(size_t)p * N_in + i] = out_Z[(size_t)h.h1_idx * N_in + i] * out_Z[(size_t)h.h2_idx * N_in + i];
                } else {
                    float val = 0.0f;
                    for (int d = 0; d < D; d++) val += h.w[d] * X[i * D + d];
                    out_Z[(size_t)p * N_in + i] = apply_op(val, h.type);
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
    std::vector<float>   split_gain;       // [total_nodes], actual gain per split node (0 for leaves)
};

// ── Histogram BFS tree build (internal, used by pool_build_tree) ─────────────
//
// 255-bin uniform histogram split search.
// Cost per hypothesis per node: O(Ns + 255*K)  vs  old O(9 * Ns * K)
// At K=19, Ns=25000: ~55k ops vs ~4.3M → ~78× faster per split search.

static const int N_HIST_BINS = 255;

static BFSTree* bfs_build_hist(
    const HypForgePool* pool,
    const float* Z_sub,    // [P, Ns], subsampled projections
    const float* G,        // [Ns, K]
    const float* H,        // [Ns, K]
    int P, int Ns, int K,
    int max_depth, int min_split, int min_leaf, float reg_lambda
) {
    BFSTree* tree     = new BFSTree();
    tree->max_depth   = max_depth;
    tree->K           = K;
    int total_nodes   = (1 << (max_depth + 1)) - 1;
    tree->total_nodes = total_nodes;
    tree->split_hyp_idx.assign(total_nodes, -1);
    tree->split_threshold.assign(total_nodes, 0.0f);
    tree->split_gain.assign(total_nodes, 0.0f);
    tree->leaf_values.assign((size_t)total_nodes * K, 0.0f);
    tree->is_leaf.assign(total_nodes, 1);

    std::vector<std::vector<int>> node_samples(total_nodes);
    node_samples[0].resize(Ns);
    std::iota(node_samples[0].begin(), node_samples[0].end(), 0);

    std::vector<float> G_sum(K), H_sum(K);
    std::vector<float> G_tot(K), H_tot(K);

    for (int depth = 0; depth <= max_depth; depth++) {
        int n_nodes  = 1 << depth;
        int base     = n_nodes - 1;
        bool is_last = (depth == max_depth);

        for (int local = 0; local < n_nodes; local++) {
            int t         = base + local;
            auto& samples = node_samples[t];
            int  n_in     = (int)samples.size();

            if (n_in == 0) { tree->is_leaf[t] = 1; continue; }

            // ── Leaf value ───────────────────────────────────────────────────
            std::fill(G_sum.begin(), G_sum.end(), 0.0f);
            std::fill(H_sum.begin(), H_sum.end(), 0.0f);
            for (int idx : samples) {
                const float* gi = G + (size_t)idx * K;
                const float* hi = H + (size_t)idx * K;
                for (int k = 0; k < K; k++) { G_sum[k] += gi[k]; H_sum[k] += hi[k]; }
            }
            float* lv = &tree->leaf_values[(size_t)t * K];
            for (int k = 0; k < K; k++) lv[k] = -(G_sum[k] / (H_sum[k] + reg_lambda));

            if (is_last || n_in < min_split) {
                tree->is_leaf[t] = 1;
                { std::vector<int> tmp; tmp.swap(samples); }
                continue;
            }

            // ── Subsample for split search ───────────────────────────────────
            const int* sp = samples.data();
            int Ns_node   = n_in;
            std::vector<int> sub_buf;
            if (n_in > 25000) {
                Ns_node = 25000;
                sub_buf.assign(samples.begin(), samples.begin() + Ns_node);
                sp = sub_buf.data();
            }

            std::fill(G_tot.begin(), G_tot.end(), 0.0f);
            std::fill(H_tot.begin(), H_tot.end(), 0.0f);
            for (int j = 0; j < Ns_node; j++) {
                const float* gi = G + (size_t)sp[j] * K;
                const float* hi = H + (size_t)sp[j] * K;
                for (int k = 0; k < K; k++) { G_tot[k] += gi[k]; H_tot[k] += hi[k]; }
            }
            float parent_score = 0.0f;
            for (int k = 0; k < K; k++)
                parent_score += G_tot[k] * G_tot[k] / (G_tot[k] > 0 ? H_tot[k] + reg_lambda : H_tot[k] + reg_lambda);
            // (simplified: parent_score = sum G^2/(H+lam))
            parent_score = 0.0f;
            for (int k = 0; k < K; k++)
                parent_score += G_tot[k] * G_tot[k] / (H_tot[k] + reg_lambda);

            // ── Histogram split search (parallelised over hypotheses) ────────
            float best_gain = 1e-5f;
            int   best_p    = -1;
            float best_thr  = 0.0f;

#ifdef _OPENMP
            #pragma omp parallel
            {
                std::vector<float> G_hist(N_HIST_BINS * K);
                std::vector<float> H_hist(N_HIST_BINS * K);
                std::vector<int>   cnt_hist(N_HIST_BINS);
                std::vector<float> G_L(K), H_L(K);
                float tl_gain = 1e-5f; int tl_p = -1; float tl_thr = 0.0f;

                #pragma omp for nowait
                for (int p = 0; p < P; p++) {
                    float bmin = pool->pop[p].bin_min;
                    float bmax = pool->pop[p].bin_max;
                    float range = bmax - bmin;
                    if (range < 1e-8f) continue;
                    float inv_r = N_HIST_BINS / range;

                    const float* Z_p = Z_sub + (size_t)p * Ns;

                    std::fill(G_hist.begin(), G_hist.end(), 0.0f);
                    std::fill(H_hist.begin(), H_hist.end(), 0.0f);
                    std::fill(cnt_hist.begin(), cnt_hist.end(), 0);

                    for (int j = 0; j < Ns_node; j++) {
                        float z = Z_p[sp[j]];
                        int b = (int)((z - bmin) * inv_r);
                        if (b < 0) b = 0;
                        if (b >= N_HIST_BINS) b = N_HIST_BINS - 1;
                        cnt_hist[b]++;
                        const float* gi = G + (size_t)sp[j] * K;
                        const float* hi = H + (size_t)sp[j] * K;
                        for (int k = 0; k < K; k++) {
                            G_hist[b * K + k] += gi[k];
                            H_hist[b * K + k] += hi[k];
                        }
                    }

                    std::fill(G_L.begin(), G_L.end(), 0.0f);
                    std::fill(H_L.begin(), H_L.end(), 0.0f);
                    int n_left = 0;
                    for (int b = 0; b < N_HIST_BINS - 1; b++) {
                        n_left += cnt_hist[b];
                        for (int k = 0; k < K; k++) {
                            G_L[k] += G_hist[b * K + k];
                            H_L[k] += H_hist[b * K + k];
                        }
                        int n_right = Ns_node - n_left;
                        if (n_left < min_leaf || n_right < min_leaf) continue;

                        float gain = 0.0f;
                        for (int k = 0; k < K; k++) {
                            float GR = G_tot[k] - G_L[k];
                            float HR = H_tot[k] - H_L[k];
                            gain += G_L[k]*G_L[k]/(H_L[k]+reg_lambda)
                                  + GR*GR/(HR+reg_lambda);
                        }
                        gain = 0.5f * (gain - parent_score);
                        float split_val = bmin + (b + 1) / inv_r;
                        if (gain > tl_gain) { tl_gain = gain; tl_p = p; tl_thr = split_val; }
                    }
                }
                #pragma omp critical
                { if (tl_gain > best_gain) { best_gain = tl_gain; best_p = tl_p; best_thr = tl_thr; } }
            }
#else
            {
                std::vector<float> G_hist(N_HIST_BINS * K);
                std::vector<float> H_hist(N_HIST_BINS * K);
                std::vector<int>   cnt_hist(N_HIST_BINS);
                std::vector<float> G_L(K), H_L(K);

                for (int p = 0; p < P; p++) {
                    float bmin = pool->pop[p].bin_min;
                    float bmax = pool->pop[p].bin_max;
                    float range = bmax - bmin;
                    if (range < 1e-8f) continue;
                    float inv_r = N_HIST_BINS / range;

                    const float* Z_p = Z_sub + (size_t)p * Ns;

                    std::fill(G_hist.begin(), G_hist.end(), 0.0f);
                    std::fill(H_hist.begin(), H_hist.end(), 0.0f);
                    std::fill(cnt_hist.begin(), cnt_hist.end(), 0);

                    for (int j = 0; j < Ns_node; j++) {
                        float z = Z_p[sp[j]];
                        int b = (int)((z - bmin) * inv_r);
                        if (b < 0) b = 0;
                        if (b >= N_HIST_BINS) b = N_HIST_BINS - 1;
                        cnt_hist[b]++;
                        const float* gi = G + (size_t)sp[j] * K;
                        const float* hi = H + (size_t)sp[j] * K;
                        for (int k = 0; k < K; k++) {
                            G_hist[b * K + k] += gi[k];
                            H_hist[b * K + k] += hi[k];
                        }
                    }

                    std::fill(G_L.begin(), G_L.end(), 0.0f);
                    std::fill(H_L.begin(), H_L.end(), 0.0f);
                    int n_left = 0;
                    for (int b = 0; b < N_HIST_BINS - 1; b++) {
                        n_left += cnt_hist[b];
                        for (int k = 0; k < K; k++) {
                            G_L[k] += G_hist[b * K + k];
                            H_L[k] += H_hist[b * K + k];
                        }
                        int n_right = Ns_node - n_left;
                        if (n_left < min_leaf || n_right < min_leaf) continue;

                        float gain = 0.0f;
                        for (int k = 0; k < K; k++) {
                            float GR = G_tot[k] - G_L[k];
                            float HR = H_tot[k] - H_L[k];
                            gain += G_L[k]*G_L[k]/(H_L[k]+reg_lambda)
                                  + GR*GR/(HR+reg_lambda);
                        }
                        gain = 0.5f * (gain - parent_score);
                        float split_val = bmin + (b + 1) / inv_r;
                        if (gain > best_gain) { best_gain = gain; best_p = p; best_thr = split_val; }
                    }
                }
            }
#endif

            if (best_p < 0) {
                tree->is_leaf[t] = 1;
                { std::vector<int> tmp; tmp.swap(samples); }
                continue;
            }

            // ── Record split & route ─────────────────────────────────────────
            tree->is_leaf[t]         = 0;
            tree->split_hyp_idx[t]   = best_p;
            tree->split_threshold[t] = best_thr;
            tree->split_gain[t]      = best_gain;

            int t_left  = 2 * t + 1;
            int t_right = 2 * t + 2;
            const float* Z_best = Z_sub + (size_t)best_p * Ns;
            node_samples[t_left ].reserve(n_in / 2 + 8);
            node_samples[t_right].reserve(n_in / 2 + 8);
            for (int idx : samples) {
                if (Z_best[idx] < best_thr) node_samples[t_left ].push_back(idx);
                else                        node_samples[t_right].push_back(idx);
            }
            { std::vector<int> tmp; tmp.swap(samples); }
        }
    }
    return tree;
}

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
    tree->split_gain.assign(total_nodes, 0.0f);

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
            tree->split_gain[t]      = best_gain;

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

void* pool_create(int D, int max_size, int evolve_mode) {
    auto* pool = new HypForgePool(D, max_size);
    return static_cast<void*>(pool);
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

void pool_set_options(
    void* handle,
    int op_mode, int crossover_top_k, int elitism_k, int alps_mode, int map_elites_slots
) {
    auto* pool = static_cast<HypForgePool*>(handle);
    pool->op_mode           = op_mode;
    pool->crossover_top_k   = crossover_top_k;
    pool->elitism_k         = elitism_k;
    pool->alps_mode         = alps_mode;
    pool->map_elites_slots  = map_elites_slots;
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
        
        if (h.type < 2) {
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
        
        if (h.type < 2) {
            h.w.assign(weights + p * D, weights + (p + 1) * D);
        }
        h.update_complexity(pool->pop);
        pool->pop.push_back(std::move(h));
    }
    return static_cast<void*>(pool);
}

// ── Integrated: evolve + build + predict in one C++ call ─────────────────────
//
// Replaces the Python sequence:
//   pool.evolve() → get_caches_and_thresholds() → bfstree_build() →
//   tree.predict() → update_use_counts()
//
// Benefits:
//   • Z_full [P×N] never allocated in Python (saves ~175 MB per round)
//   • Prediction uses full_cache directly, avoiding a second Z_full copy
//   • use_count update from actual tree splits (not deferred)
//
// Returns a BFSTree handle (same ownership semantics as bfstree_build).
void* pool_build_tree(
    void*  pool_handle,
    const float* X,          // [N, D]
    const float* G_full,     // [N, K]
    const float* H_full,     // [N, K]
    int N, int K,
    const int* evolve_sub,   int Ns_evolve,
    const int* tree_sub,     int Ns_tree,
    int D_num, int max_depth,
    float reg_lambda, float eta_penalty,
    int do_evolve,           // 1 = run evolve, 0 = skip (reuse last state)
    float* out_pred          // [N, K], caller allocs, caller zero-inits
) {
    auto* pool = static_cast<HypForgePool*>(pool_handle);

    // ── 1. Optionally evolve pool ─────────────────────────────────────────────
    if (do_evolve) {
        pool->evolve(X, G_full, H_full, evolve_sub,
                     N, Ns_evolve, K, D_num, reg_lambda, eta_penalty);
    }

    int P = (int)pool->pop.size();
    if (P == 0) return nullptr;

    // ── 2. Build Z_sub [P, Ns_tree] from full_cache using persistent buffer ───
    //    Reusing pool->Z_buf_ / G_buf_ / H_buf_ avoids ~30-50 MB malloc per round.
    size_t z_need = (size_t)P * Ns_tree;
    size_t gh_need = (size_t)Ns_tree * K;
    if (pool->Z_buf_.size() < z_need)  pool->Z_buf_.resize(z_need);
    if (pool->G_buf_.size() < gh_need) pool->G_buf_.resize(gh_need);
    if (pool->H_buf_.size() < gh_need) pool->H_buf_.resize(gh_need);

    float* Z_sub = pool->Z_buf_.data();
    float* G_sub = pool->G_buf_.data();
    float* H_sub = pool->H_buf_.data();

    for (int p = 0; p < P; p++) {
        const float* fc = pool->pop[p].full_cache.data();
        float* Zp = Z_sub + (size_t)p * Ns_tree;
        for (int j = 0; j < Ns_tree; j++) Zp[j] = fc[tree_sub[j]];
    }

    // ── 3. Subsample G, H for tree build ─────────────────────────────────────
    for (int j = 0; j < Ns_tree; j++) {
        int idx = tree_sub[j];
        std::memcpy(G_sub + (size_t)j * K, G_full + (size_t)idx * K, K * sizeof(float));
        std::memcpy(H_sub + (size_t)j * K, H_full + (size_t)idx * K, K * sizeof(float));
    }

    // ── 4. Build BFS tree using 255-bin histogram splits ─────────────────────
    //    bfs_build_hist uses pool->pop[p].bin_min/bin_max for uniform binning.
    //    Split search cost: O(Ns + 255*K) vs old O(9*Ns*K) ≈ 78× faster at K=19.
    BFSTree* tree = bfs_build_hist(
        pool,
        Z_sub, G_sub, H_sub,
        P, Ns_tree, K,
        max_depth, 20, 10, reg_lambda
    );

    // ── 5. Update use_counts from actual tree splits ─────────────────────────
    {
        int n_nodes = tree->total_nodes;
        for (int p = 0; p < P; p++) pool->pop[p].rounds_since_last_use++;
        for (int t = 0; t < n_nodes; t++) {
            int idx = tree->split_hyp_idx[t];
            if (idx >= 0 && idx < P) {
                pool->pop[idx].use_count++;
                pool->pop[idx].rounds_since_last_use = 0;
            }
        }
    }

    // ── 6. Predict on full N using full_cache — no Z_full allocation ──────────
    // Walk each sample through tree routing; full_cache[p][i] replaces Z[p*N+i].
    {
        std::vector<int> node_assign(N, 0);
        int max_d = tree->max_depth;

        for (int depth = 0; depth < max_d; depth++) {
            int n_nodes = 1 << depth;
            int base    = n_nodes - 1;
            for (int local = 0; local < n_nodes; local++) {
                int t = base + local;
                if (tree->is_leaf[t]) continue;
                int   p      = tree->split_hyp_idx[t];
                float thr    = tree->split_threshold[t];
                const float* fc = pool->pop[p].full_cache.data();
                int t_left = 2 * t + 1, t_right = 2 * t + 2;
                for (int i = 0; i < N; i++) {
                    if (node_assign[i] == t)
                        node_assign[i] = (fc[i] < thr) ? t_left : t_right;
                }
            }
        }

        for (int i = 0; i < N; i++) {
            const float* lv = tree->leaf_values.data() + (size_t)node_assign[i] * K;
            float* oi = out_pred + (size_t)i * K;
            for (int k = 0; k < K; k++) oi[k] = lv[k];
        }
    }

    return static_cast<void*>(tree);
}

} // extern "C"

