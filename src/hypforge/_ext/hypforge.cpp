// hypforge.cpp — HypForge Pool Engine (UCB bandit pool + hypothesis evolution)
//
// Compiled together with bfstree.cpp into one shared library.
// Tree structures and BFSTree C API live in bfstree.cpp.
// Pool C API (pool_create, pool_evolve, pool_build_tree, …) lives here.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "bfstree_types.h"

// ── Operator application ─────────────────────────────────────────────────────
//   0 = linear     : w·x
//   1 = leaky_relu : max(w·x, 0.01·w·x)
//   2 = product    : h1·h2  (composite)
static inline float apply_op(float val, int type) {
  if (type == 1) return (val > 0.0f) ? val : 0.01f * val;
  return val;
}

// ── Hypothesis ───────────────────────────────────────────────────────────────
struct Hypothesis {
  int type = 0;
  std::vector<float> w;
  int h1_idx = -1;
  int h2_idx = -1;

  // bandit stats
  int n_obs = 0;
  double mu_fitness = 0.0;
  double M2_fitness = 0.0;

  int use_count = 0;
  int rounds_since_last_use = 0;

  float fitness = 0.0f;
  float score = 0.0f;
  bool is_base = false;

  // genealogy
  int global_id = -1;
  int parent1 = -1;
  int parent2 = -1;
  int birth_round = 0;
  int family_id = -1;
  float family_fitness = 0.0f;
  float breeding_value = 0.0f;
  float ancestor_credit = 0.0f;

  // projection caches
  std::vector<float> full_cache;  // size N
  std::vector<float> thresholds;  // size 9
  float bin_min = 0.0f;
  float bin_max = 1.0f;
  int stored_complexity = 0;

  int complexity() const { return stored_complexity; }

  void update_complexity(const std::vector<Hypothesis>& pool) {
    if (type == 2) {
      stored_complexity = pool[h1_idx].complexity() + pool[h2_idx].complexity();
    } else {
      stored_complexity = 0;
      for (float v : w)
        if (std::abs(v) > 1e-3f) stored_complexity++;
    }
  }

  float ucb_score(float eta_penalty) const {
    if (n_obs == 0) return 0.0f;
    float n = (float)n_obs;
    float sigma = (n_obs > 1) ? std::sqrt((float)(M2_fitness / n)) : 0.0f;
    float bonus = (sigma + 0.5f) / std::sqrt(n);
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

// ── Meta-Evolution policy ────────────────────────────────────────────────────
struct EvolutionPolicy {
  int crossover_top_k;
  float family_lambda;
  float breeding_beta;
  int use_count = 0;
  double sum_reward = 0.0;
  double mean_reward = 0.0;
};

// ── Helpers ──────────────────────────────────────────────────────────────────
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

// ── HypForgePool ─────────────────────────────────────────────────────────────
class HypForgePool {
 public:
  int D;
  int max_size;
  int N = 0;
  int op_mode = 0;
  int crossover_top_k = 6;
  int elitism_k = 0;
  int alps_mode = 0;
  int map_elites_slots = 0;
  int family_max_size = 30;
  int enable_meta_evolution = 1;
  float user_family_lambda = 0.1f;
  float user_breeding_beta = 0.3f;

  std::vector<Hypothesis> pop;
  std::vector<Hypothesis> history;
  float cx_births[3][3];
  float cx_survivors[3][3];

  std::vector<EvolutionPolicy> policies;
  int active_policy_idx = 0;
  int total_policy_rounds = 0;
  unsigned int seed = 42;

  // Persistent scratch buffers
  std::vector<float> Z_buf_;
  std::vector<float> G_buf_;
  std::vector<float> H_buf_;

  HypForgePool(int D, int max_size) : D(D), max_size(max_size) {
    std::memset(cx_births, 0, sizeof(cx_births));
    std::memset(cx_survivors, 0, sizeof(cx_survivors));

    policies.push_back({6, 0.1f, 0.3f});
    policies.push_back({3, 0.05f, 0.5f});
    policies.push_back({10, 0.3f, 0.1f});
    policies.push_back({4, 0.1f, 0.6f});

    for (int j = 0; j < D; j++) {
      Hypothesis h;
      h.type = 0;
      h.w.assign(D, 0.0f);
      h.w[j] = 1.0f;
      h.is_base = true;
      h.global_id = (int)history.size();
      h.family_id = j;
      h.update_complexity(pop);

      Hypothesis hist_h = h;
      hist_h.full_cache.clear();
      hist_h.full_cache.shrink_to_fit();
      hist_h.thresholds.clear();
      hist_h.thresholds.shrink_to_fit();
      history.push_back(std::move(hist_h));
      pop.push_back(std::move(h));
    }
  }

  void propagate_credit(int gid, float gain, float gamma, int depth) {
    if (gid < 0 || gid >= (int)history.size()) return;
    history[gid].ancestor_credit += gain * std::pow(gamma, depth);
    int p1 = history[gid].parent1;
    int p2 = history[gid].parent2;
    if (p1 >= 0) propagate_credit(p1, gain, gamma, depth + 1);
    if (p2 >= 0) propagate_credit(p2, gain, gamma, depth + 1);
  }

  void update_genealogy_stats() {}  // replaced by credit propagation

  std::vector<float> get_cache_for_history_node(int gid, const float* X) const {
    std::vector<float> cache(N, 0.0f);
    if (gid < 0 || gid >= (int)history.size()) return cache;
    const auto& h = history[gid];
    if (h.type == 2) {
      auto c1 = get_cache_for_history_node(h.h1_idx, X);
      auto c2 = get_cache_for_history_node(h.h2_idx, X);
      for (int i = 0; i < N; i++) cache[i] = c1[i] * c2[i];
    } else {
      for (int i = 0; i < N; i++) {
        float val = 0.0f;
        for (int d = 0; d < D; d++) val += h.w[d] * X[i * D + d];
        cache[i] = apply_op(val, h.type);
      }
    }
    return cache;
  }

  void eval_hypothesis(const Hypothesis& h, const float* X, int N_in,
                       float* out) const {
    if (h.type == 2) {
      std::vector<float> t1(N_in, 0.0f), t2(N_in, 0.0f);
      if (h.h1_idx >= 0 && h.h1_idx < (int)history.size())
        eval_hypothesis(history[h.h1_idx], X, N_in, t1.data());
      if (h.h2_idx >= 0 && h.h2_idx < (int)history.size())
        eval_hypothesis(history[h.h2_idx], X, N_in, t2.data());
      for (int i = 0; i < N_in; i++) out[i] = t1[i] * t2[i];
    } else {
      for (int i = 0; i < N_in; i++) {
        float val = 0.0f;
        for (int d = 0; d < D; d++) val += h.w[d] * X[i * D + d];
        out[i] = apply_op(val, h.type);
      }
    }
  }

  void initialize_cache(Hypothesis& h, const float* X,
                        const std::vector<int>& rand_indices) {
    h.full_cache.resize(N);
    if (h.type == 2) {
      auto c1 = get_cache_for_history_node(h.h1_idx, X);
      auto c2 = get_cache_for_history_node(h.h2_idx, X);
      for (int i = 0; i < N; i++) h.full_cache[i] = c1[i] * c2[i];
    } else {
      for (int i = 0; i < N; i++) {
        float val = 0.0f;
        for (int d = 0; d < D; d++) val += h.w[d] * X[i * D + d];
        h.full_cache[i] = apply_op(val, h.type);
      }
    }

    int Ns_q = (int)rand_indices.size();
    std::vector<float> sample(Ns_q);
    for (int i = 0; i < Ns_q; i++) sample[i] = h.full_cache[rand_indices[i]];
    std::sort(sample.begin(), sample.end());

    h.thresholds.resize(9);
    for (int q = 0; q < 9; q++) {
      int idx = (int)((q + 1) * 0.1f * Ns_q);
      if (idx >= Ns_q) idx = Ns_q - 1;
      h.thresholds[q] = sample[idx];
    }
    h.bin_min = sample.front();
    h.bin_max = sample.back();
  }

  std::vector<float> sparsify(const std::vector<float>& w, int D_num) {
    int D_in = (int)w.size();
    std::vector<float> w_sp = w;
    if (D_num < D_in)
      for (int i = D_num; i < D_in; i++) w_sp[i] = 0.0f;

    int k = std::max(2, (int)std::sqrt((float)D_num));
    if (D_num > k) {
      std::vector<std::pair<float, int>> abs_vals(D_num);
      for (int i = 0; i < D_num; i++) abs_vals[i] = {std::abs(w_sp[i]), i};
      std::sort(abs_vals.begin(), abs_vals.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
      std::vector<bool> keep(D_num, false);
      for (int i = 0; i < k; i++) keep[abs_vals[i].second] = true;
      for (int i = 0; i < D_num; i++)
        if (!keep[i]) w_sp[i] = 0.0f;
    }

    float norm = 0.0f;
    for (float v : w_sp) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-5f) {
      for (float& v : w_sp) v /= norm;
      return w_sp;
    }
    std::vector<float> fallback(D_in, 0.0f);
    std::mt19937 g(seed++);
    fallback[g() % D_num] = 1.0f;
    return fallback;
  }

  void evolve(const float* X, const float* G_full, const float* H_full,
              const int* sub_indices, int N_in, int Ns, int K, int D_num,
              float reg_lambda, float eta_penalty = 0.002f,
              int current_round = 0) {
    this->N = N_in;
    int min_leaf = std::max(1, std::min(10, Ns / 5));
    std::vector<int> rand_indices =
        get_random_indices(N, std::min(N, 10000), seed++);
    for (auto& h : pop)
      if (h.full_cache.empty()) initialize_cache(h, X, rand_indices);

    float family_lambda = (enable_meta_evolution > 0 && !policies.empty())
                              ? policies[active_policy_idx].family_lambda
                              : user_family_lambda;
    float breeding_beta = (enable_meta_evolution > 0 && !policies.empty())
                              ? policies[active_policy_idx].breeding_beta
                              : user_breeding_beta;
    int cx_top_k = (enable_meta_evolution > 0 && !policies.empty())
                       ? policies[active_policy_idx].crossover_top_k
                       : crossover_top_k;

    // ── Fitness scan (existing pop) ──────────────────────────────────────────
    std::vector<float> G_tot(K, 0.0f), H_tot(K, 0.0f);
    for (int j = 0; j < Ns; j++) {
      int idx = sub_indices[j];
      for (int k = 0; k < K; k++) {
        G_tot[k] += G_full[idx * K + k];
        H_tot[k] += H_full[idx * K + k];
      }
    }
    float parent_score = 0.0f;
    for (int k = 0; k < K; k++)
      parent_score += G_tot[k] * G_tot[k] / (H_tot[k] + reg_lambda);

    int P = (int)pop.size();
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
        if (n_left < min_leaf || n_right < min_leaf) continue;
        float gain = 0.0f;
        for (int k = 0; k < K; k++) {
          float GR = G_tot[k] - G_L[k], HR = H_tot[k] - H_L[k];
          gain += G_L[k] * G_L[k] / (H_L[k] + reg_lambda) +
                  GR * GR / (HR + reg_lambda);
        }
        gain = 0.5f * (gain - parent_score);
        if (gain > best_gain) best_gain = gain;
      }
      best_gains[p] = best_gain;
    }
    for (int p = 0; p < P; p++)
      pop[p].observe_fitness(best_gains[p], eta_penalty);

    // Sync to history
    for (int p = 0; p < P; p++) {
      int gid = pop[p].global_id;
      if (gid >= 0 && gid < (int)history.size()) {
        history[gid].fitness = pop[p].fitness;
        history[gid].n_obs = pop[p].n_obs;
        history[gid].mu_fitness = pop[p].mu_fitness;
        history[gid].M2_fitness = pop[p].M2_fitness;
        history[gid].score = pop[p].score;
      }
    }
    update_genealogy_stats();
    for (auto& h : pop) {
      int gid = h.global_id;
      if (gid >= 0 && gid < (int)history.size()) {
        h.ancestor_credit = history[gid].ancestor_credit;
        h.family_fitness = history[gid].ancestor_credit;
        h.breeding_value = history[gid].ancestor_credit;
      }
      h.score = h.ucb_score(eta_penalty) + family_lambda * h.ancestor_credit;
    }
    std::sort(pop.begin(), pop.end(),
              [](const Hypothesis& a, const Hypothesis& b) {
                return a.score > b.score;
              });

    // ── Generate candidates ──────────────────────────────────────────────────
    std::vector<Hypothesis> raw_candidates;

    // Gradient-aligned candidates
    std::vector<float> V(D * K, 0.0f);
    for (int d = 0; d < D; d++)
      for (int k = 0; k < K; k++) {
        float sv = 0.0f;
        for (int j = 0; j < Ns; j++) {
          int idx = sub_indices[j];
          sv += X[idx * D + d] * G_full[idx * K + k];
        }
        V[d * K + k] = sv;
      }
    for (int k = 0; k < K; k++) {
      std::vector<float> V_k(D);
      for (int d = 0; d < D; d++) V_k[d] = V[d * K + k];
      std::vector<float> w_k = sparsify(V_k, D_num);
      int type_lo = (op_mode == 2) ? 1 : 0;
      int type_hi = (op_mode == 1) ? 1 : 2;
      for (int type = type_lo; type < type_hi; type++) {
        Hypothesis c;
        c.type = type;
        c.w = w_k;
        c.parent1 = -1;
        c.parent2 = -1;
        c.birth_round = current_round;
        c.family_id = -1;
        c.update_complexity(pop);
        raw_candidates.push_back(c);
      }
    }

    // Crossover
    if ((int)pop.size() >= 2) {
      struct ParentInfo {
        int pop_idx;
        float parent_score;
      };
      std::vector<ParentInfo> parents;
      for (int p = 0; p < (int)pop.size(); p++)
        parents.push_back({p, pop[p].fitness + pop[p].ancestor_credit});
      std::sort(parents.begin(), parents.end(),
                [](const ParentInfo& a, const ParentInfo& b) {
                  return a.parent_score > b.parent_score;
                });

      int n_top = std::min((int)parents.size(), cx_top_k);
      if (n_top >= 2) {
        struct PairInfo {
          int p1_idx, p2_idx;
          float weight;
        };
        std::vector<PairInfo> possible_pairs;
        double total_weight = 0.0;
        for (int i = 0; i < n_top; i++) {
          int idx_i = parents[i].pop_idx;
          if (pop[idx_i].type >= 2) continue;
          for (int j = i + 1; j < n_top; j++) {
            int idx_j = parents[j].pop_idx;
            if (pop[idx_j].type >= 2) continue;
            possible_pairs.push_back({idx_i, idx_j, 1.0f});
            total_weight += 1.0;
          }
        }
        if (!possible_pairs.empty() && total_weight > 0.0) {
          std::mt19937 rng_cx(seed++);
          std::uniform_real_distribution<double> dist_prob(0.0, total_weight);
          std::uniform_real_distribution<float> alpha_dist(0.3f, 0.7f);
          int num_to_sample = (int)possible_pairs.size();
          for (int s = 0; s < num_to_sample; s++) {
            double r = dist_prob(rng_cx), running = 0.0;
            PairInfo sel = possible_pairs[0];
            for (const auto& pair : possible_pairs) {
              running += pair.weight;
              if (r <= running) {
                sel = pair;
                break;
              }
            }
            const auto& pa = pop[sel.p1_idx];
            const auto& pb = pop[sel.p2_idx];
            float alpha = alpha_dist(rng_cx);
            std::vector<float> w_child(D);
            for (int d = 0; d < D; d++)
              w_child[d] = alpha * pa.w[d] + (1.0f - alpha) * pb.w[d];
            w_child = sparsify(w_child, D_num);
            Hypothesis c;
            c.type = (pa.score >= pb.score) ? pa.type : pb.type;
            c.w = w_child;
            c.parent1 = pa.global_id;
            c.parent2 = pb.global_id;
            c.birth_round = current_round;
            c.family_id = (pa.score >= pb.score) ? pa.family_id : pb.family_id;
            c.update_complexity(pop);
            raw_candidates.push_back(c);
          }
        }
      }
    }

    if (raw_candidates.empty()) return;

    // ── Candidate fitness scan ───────────────────────────────────────────────
    int C_cand = (int)raw_candidates.size();
    std::vector<std::vector<float>> cand_sub_cache(C_cand,
                                                   std::vector<float>(Ns));
    std::vector<float> cand_best_gain(C_cand, 0.0f);
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
        if (n_left < min_leaf || n_right < min_leaf) continue;
        float gain = 0.0f;
        for (int k = 0; k < K; k++) {
          float GR = G_tot[k] - G_L[k], HR = H_tot[k] - H_L[k];
          gain += G_L[k] * G_L[k] / (H_L[k] + reg_lambda) +
                  GR * GR / (HR + reg_lambda);
        }
        gain = 0.5f * (gain - parent_score);
        if (gain > best_gain) best_gain = gain;
      }
      cand_best_gain[ci] = best_gain;
    }

    // Admit candidates with positive gain
    std::vector<Hypothesis> admitted;
    for (int ci = 0; ci < C_cand; ci++) {
      if (cand_best_gain[ci] > 1e-5f) {
        auto& c = raw_candidates[ci];
        c.observe_fitness(cand_best_gain[ci], eta_penalty);
        c.global_id = (int)history.size();
        if (c.family_id == -1) c.family_id = c.global_id;
        Hypothesis hist_c = c;
        hist_c.full_cache.clear();
        hist_c.full_cache.shrink_to_fit();
        hist_c.thresholds.clear();
        hist_c.thresholds.shrink_to_fit();
        history.push_back(std::move(hist_c));
        admitted.push_back(c);
      }
    }

    // Track births for Operator Transition Matrix
    for (const auto& c : admitted) {
      if (c.birth_round == current_round && c.parent1 != -1 &&
          c.parent2 != -1 && c.type < 2) {
        int p1t = history[c.parent1].type, p2t = history[c.parent2].type;
        if (p1t >= 0 && p1t < 3 && p2t >= 0 && p2t < 3) {
          cx_births[p1t][p2t]++;
          if (p1t != p2t) cx_births[p2t][p1t]++;
        }
      }
    }

    for (auto& c : admitted) {
      initialize_cache(c, X, rand_indices);
      pop.push_back(c);
    }

    // Re-score & sort
    for (auto& h : pop) {
      int gid = h.global_id;
      float fam_fit = (gid >= 0 && gid < (int)history.size())
                          ? history[gid].family_fitness
                          : h.fitness;
      h.family_fitness = fam_fit;
      h.score = h.ucb_score(eta_penalty) + family_lambda * fam_fit;
    }
    std::sort(pop.begin(), pop.end(),
              [](const Hypothesis& a, const Hypothesis& b) {
                return a.score > b.score;
              });

    P = (int)pop.size();

    // ── Similarity dedup ─────────────────────────────────────────────────────
    int Ns_sim = std::min(Ns, 1000);
    std::vector<float> sub_caches_flat((size_t)P * Ns_sim);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int p = 0; p < P; p++) {
      float norm_sq = 0.0f;
      for (int j = 0; j < Ns_sim; j++) {
        float val = pop[p].full_cache[sub_indices[j]];
        sub_caches_flat[(size_t)p * Ns_sim + j] = val;
        norm_sq += val * val;
      }
      float norm = std::sqrt(norm_sq) + 1e-8f;
      for (int j = 0; j < Ns_sim; j++)
        sub_caches_flat[(size_t)p * Ns_sim + j] /= norm;
    }

    std::vector<float> sim_matrix((size_t)P * P, 0.0f);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 8)
#endif
    for (int i = 0; i < P; i++) {
      for (int j = i + 1; j < P; j++) {
        float dot = 0.0f;
        const float* ci = &sub_caches_flat[(size_t)i * Ns_sim];
        const float* cj = &sub_caches_flat[(size_t)j * Ns_sim];
        for (int s = 0; s < Ns_sim; s++) dot += ci[s] * cj[s];
        float sim = std::abs(dot);
        sim_matrix[(size_t)i * P + j] = sim;
        sim_matrix[(size_t)j * P + i] = sim;
      }
    }

    // ALPS priority boost
    if (alps_mode > 0) {
      for (auto& h : pop)
        if (h.n_obs < 5) h.score += 1e6f;
      std::sort(pop.begin(), pop.end(),
                [](const Hypothesis& a, const Hypothesis& b) {
                  return a.score > b.score;
                });
      for (auto& h : pop)
        if (h.n_obs < 5) h.score -= 1e6f;
    }

    // Elitism + base always kept
    std::vector<bool> in_kept(P, false);
    std::vector<int> kept;
    int elite_cap = (elitism_k > 0) ? std::min(elitism_k, P) : 0;
    for (int i = 0; i < elite_cap; i++) {
      kept.push_back(i);
      in_kept[i] = true;
    }
    for (int i = 0; i < P; i++) {
      if (pop[i].is_base && !in_kept[i]) {
        kept.push_back(i);
        in_kept[i] = true;
      }
    }

    std::unordered_map<int, int> family_counts;
    for (int idx : kept) family_counts[pop[idx].family_id]++;

    for (int i = 0; i < P; i++) {
      if (i < elite_cap || pop[i].is_base) continue;
      int fid = pop[i].family_id;
      if (family_max_size > 0 && family_counts[fid] >= family_max_size)
        continue;
      bool redundant = false;
      for (int j : kept) {
        float thr_val = (pop[i].type == pop[j].type) ? 0.90f : 0.98f;
        if (sim_matrix[(size_t)i * P + j] > thr_val) {
          redundant = true;
          break;
        }
      }
      if (!redundant) {
        kept.push_back(i);
        in_kept[i] = true;
        family_counts[fid]++;
        if ((int)kept.size() >= max_size) break;
      }
    }

    // MAP-Elites type quota
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

    std::vector<Hypothesis> pruned;
    for (int i : kept) pruned.push_back(std::move(pop[i]));
    pop = std::move(pruned);

    std::vector<Hypothesis> survivor_pop;
    for (auto& h : pop) {
      if (h.is_base || h.n_obs < 3 || h.mu_fitness + h.score > 1e-6f)
        survivor_pop.push_back(std::move(h));
    }
    pop = std::move(survivor_pop);

    // Track survivors
    for (const auto& h : pop) {
      if (h.birth_round == current_round && h.parent1 != -1 &&
          h.parent2 != -1 && h.type < 2) {
        int p1t = history[h.parent1].type, p2t = history[h.parent2].type;
        if (p1t >= 0 && p1t < 3 && p2t >= 0 && p2t < 3) {
          cx_survivors[p1t][p2t]++;
          if (p1t != p2t) cx_survivors[p2t][p1t]++;
        }
      }
    }
  }

  void eval(const float* X, int N_in, float* out_Z) const {
    int P_size = (int)pop.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int p = 0; p < P_size; p++)
      eval_hypothesis(pop[p], X, N_in, out_Z + (size_t)p * N_in);
  }
};

// ─────────────────────────────────────────────────────────────────────────────
// C API — Pool
// ─────────────────────────────────────────────────────────────────────────────
extern "C" {

void* pool_create(int D, int max_size, int /*evolve_mode*/) {
  return static_cast<void*>(new HypForgePool(D, max_size));
}

void pool_free(void* handle) { delete static_cast<HypForgePool*>(handle); }

void pool_set_options(void* handle, int op_mode, int crossover_top_k,
                      int elitism_k, int alps_mode, int map_elites_slots,
                      int family_max_size, int enable_meta_evolution,
                      float family_lambda, float breeding_beta) {
  auto* pool = static_cast<HypForgePool*>(handle);
  pool->op_mode = op_mode;
  pool->crossover_top_k = crossover_top_k;
  pool->elitism_k = elitism_k;
  pool->alps_mode = alps_mode;
  pool->map_elites_slots = map_elites_slots;
  pool->family_max_size = family_max_size;
  pool->enable_meta_evolution = enable_meta_evolution;
  pool->user_family_lambda = family_lambda;
  pool->user_breeding_beta = breeding_beta;
}

void pool_evolve(void* handle, const float* X, const float* G_full,
                 const float* H_full, const int* sub_indices, int N_in, int Ns,
                 int K, int D_num, float reg_lambda, float eta_penalty,
                 int current_round) {
  static_cast<HypForgePool*>(handle)->evolve(X, G_full, H_full, sub_indices,
                                             N_in, Ns, K, D_num, reg_lambda,
                                             eta_penalty, current_round);
}

void pool_eval(void* handle, const float* X, int N_in, float* out_Z) {
  static_cast<const HypForgePool*>(handle)->eval(X, N_in, out_Z);
}

int pool_get_size(void* handle) {
  return (int)static_cast<const HypForgePool*>(handle)->pop.size();
}

int pool_get_history_size(void* handle) {
  return (int)static_cast<const HypForgePool*>(handle)->history.size();
}

void pool_get_active_indices(void* handle, int* out) {
  auto* pool = static_cast<HypForgePool*>(handle);
  int P = (int)pool->pop.size();
  for (int p = 0; p < P; p++) out[p] = pool->pop[p].global_id;
}

void pool_get_caches_and_thresholds(void* handle, float* out_Z,
                                    float* out_thr) {
  auto* pool = static_cast<HypForgePool*>(handle);
  int P = (int)pool->pop.size(), N = pool->N;
  for (int p = 0; p < P; p++) {
    std::memcpy(out_Z + (size_t)p * N, pool->pop[p].full_cache.data(),
                N * sizeof(float));
    for (int q = 0; q < 9; q++) out_thr[q * P + p] = pool->pop[p].thresholds[q];
  }
}

void pool_update_use_counts(void* handle, const int* split_indices,
                            int n_nodes) {
  auto* pool = static_cast<HypForgePool*>(handle);
  int P = (int)pool->pop.size();
  for (int p = 0; p < P; p++) {
    pool->pop[p].rounds_since_last_use++;
    int gid = pool->pop[p].global_id;
    if (gid >= 0 && gid < (int)pool->history.size())
      pool->history[gid].rounds_since_last_use++;
  }
  for (int i = 0; i < n_nodes; i++) {
    int idx = split_indices[i];
    if (idx >= 0 && idx < P) {
      pool->pop[idx].use_count++;
      pool->pop[idx].rounds_since_last_use = 0;
      int gid = pool->pop[idx].global_id;
      if (gid >= 0 && gid < (int)pool->history.size()) {
        pool->history[gid].use_count++;
        pool->history[gid].rounds_since_last_use = 0;
      }
    }
  }
}

void pool_export(void* handle, int* types, float* weights, int* h1_indices,
                 int* h2_indices, int* use_counts, int* rounds_since_last_use,
                 float* credits, uint8_t* is_base, int* parent1, int* parent2,
                 int* birth_round, int* family_id, float* family_fitness,
                 float* breeding_value, float* ancestor_credit) {
  auto* pool = static_cast<HypForgePool*>(handle);
  int H = (int)pool->history.size(), D = pool->D;
  for (int p = 0; p < H; p++) {
    const auto& h = pool->history[p];
    types[p] = h.type;
    h1_indices[p] = h.h1_idx;
    h2_indices[p] = h.h2_idx;
    use_counts[p] = h.use_count;
    rounds_since_last_use[p] = h.rounds_since_last_use;
    credits[p] = h.score;
    is_base[p] = h.is_base ? 1 : 0;
    parent1[p] = h.parent1;
    parent2[p] = h.parent2;
    birth_round[p] = h.birth_round;
    family_id[p] = h.family_id;
    family_fitness[p] = h.family_fitness;
    breeding_value[p] = h.breeding_value;
    ancestor_credit[p] = h.ancestor_credit;
    if (h.type < 2)
      std::memcpy(weights + p * D, h.w.data(), D * sizeof(float));
    else
      std::memset(weights + p * D, 0, D * sizeof(float));
  }
}

void* pool_import(int D, int max_size, int U, const int* types,
                  const float* weights, const int* h1_indices,
                  const int* h2_indices, const int* use_counts,
                  const int* rounds_since_last_use, const float* credits,
                  const uint8_t* is_base, const int* parent1,
                  const int* parent2, const int* birth_round,
                  const int* family_id, const float* family_fitness,
                  const float* breeding_value, const float* ancestor_credit,
                  int P, const int* active_indices) {
  auto* pool = new HypForgePool(D, max_size);
  pool->pop.clear();
  pool->history.clear();
  for (int u = 0; u < U; u++) {
    Hypothesis h;
    h.type = types[u];
    h.h1_idx = h1_indices[u];
    h.h2_idx = h2_indices[u];
    h.use_count = use_counts[u];
    h.rounds_since_last_use = rounds_since_last_use[u];
    h.score = h.fitness = credits[u];
    h.is_base = is_base[u] != 0;
    h.parent1 = parent1[u];
    h.parent2 = parent2[u];
    h.birth_round = birth_round[u];
    h.family_id = family_id[u];
    h.family_fitness = family_fitness[u];
    h.breeding_value = breeding_value[u];
    h.ancestor_credit = ancestor_credit[u];
    h.global_id = u;
    if (h.type < 2) h.w.assign(weights + u * D, weights + (u + 1) * D);
    h.full_cache.clear();
    h.full_cache.shrink_to_fit();
    h.thresholds.clear();
    h.thresholds.shrink_to_fit();
    pool->history.push_back(std::move(h));
  }
  for (int u = 0; u < U; u++) pool->history[u].update_complexity(pool->history);
  for (int p = 0; p < P; p++) {
    int idx = active_indices[p];
    if (idx >= 0 && idx < U) pool->pop.push_back(pool->history[idx]);
  }
  return static_cast<void*>(pool);
}

void pool_get_policy_stats(void* handle, int* use_counts, double* mean_rewards,
                           int* active_idx) {
  auto* pool = static_cast<HypForgePool*>(handle);
  int n = (int)pool->policies.size();
  for (int i = 0; i < n; i++) {
    use_counts[i] = pool->policies[i].use_count;
    mean_rewards[i] = pool->policies[i].mean_reward;
  }
  *active_idx = pool->active_policy_idx;
}

void pool_get_transition_matrix(void* handle, float* births, float* survivors) {
  auto* pool = static_cast<HypForgePool*>(handle);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      births[i * 3 + j] = pool->cx_births[i][j];
      survivors[i * 3 + j] = pool->cx_survivors[i][j];
    }
}

// ── Integrated: evolve + build + predict ────────────────────────────────────
void* pool_build_tree(void* pool_handle,
                      const float* X,       // [N, D]
                      const float* G_full,  // [N, K]
                      const float* H_full,  // [N, K]
                      int N, int K, const int* evolve_sub, int Ns_evolve,
                      const int* tree_sub, int Ns_tree, int D_num,
                      int max_depth, float reg_lambda, float eta_penalty,
                      int do_evolve,
                      float* out_pred,  // [N, K], caller allocs/zero-inits
                      int current_round) {
  auto* pool = static_cast<HypForgePool*>(pool_handle);

  // Meta-Evolution: select active policy
  if (do_evolve && pool->enable_meta_evolution > 0 && !pool->policies.empty()) {
    int best_idx = 0;
    float best_ucb = -1e9f;
    float total_n = (float)pool->total_policy_rounds;
    for (int i = 0; i < (int)pool->policies.size(); i++) {
      const auto& pol = pool->policies[i];
      if (pol.use_count == 0) {
        best_idx = i;
        break;
      }
      float ucb = (float)pol.mean_reward +
                  2.0f * std::sqrt(std::log(total_n + 1.0f) / pol.use_count);
      if (ucb > best_ucb) {
        best_ucb = ucb;
        best_idx = i;
      }
    }
    pool->active_policy_idx = best_idx;
    pool->crossover_top_k = pool->policies[best_idx].crossover_top_k;
  }

  // 1. Evolve
  if (do_evolve)
    pool->evolve(X, G_full, H_full, evolve_sub, N, Ns_evolve, K, D_num,
                 reg_lambda, eta_penalty, current_round);

  int P = (int)pool->pop.size();
  if (P == 0) return nullptr;

  // 2. Build BFS tree using full caches + tree subsample indices
  std::vector<const float*> caches(P);
  for (int p = 0; p < P; p++) caches[p] = pool->pop[p].full_cache.data();
  std::vector<int> tree_indices(tree_sub, tree_sub + Ns_tree);

  BFSTree* tree =
      bfs_build_best_first(caches.data(), P, tree_indices, G_full, H_full, K,
                           max_depth, 20, 10, reg_lambda);

  // 3. Update Meta-Evolution reward
  if (do_evolve && pool->enable_meta_evolution > 0 && !pool->policies.empty()) {
    float total_gain = 0.0f;
    for (int t = 0; t < tree->total_nodes; t++)
      if (!tree->is_leaf[t]) total_gain += tree->split_gain[t];
    int idx = pool->active_policy_idx;
    auto& pol = pool->policies[idx];
    pol.use_count++;
    pol.sum_reward += total_gain;
    pol.mean_reward = pol.sum_reward / pol.use_count;
    pool->total_policy_rounds++;
  }

  // 4. Update use_counts and propagate ancestor credit
  float gamma = pool->user_breeding_beta;
  if (pool->enable_meta_evolution > 0 && !pool->policies.empty())
    gamma = pool->policies[pool->active_policy_idx].breeding_beta;

  for (int p = 0; p < P; p++) {
    pool->pop[p].rounds_since_last_use++;
    int gid = pool->pop[p].global_id;
    if (gid >= 0 && gid < (int)pool->history.size())
      pool->history[gid].rounds_since_last_use++;
  }
  for (int t = 0; t < tree->total_nodes; t++) {
    int idx = tree->split_hyp_idx[t];
    if (idx >= 0 && idx < P) {
      pool->pop[idx].use_count++;
      pool->pop[idx].rounds_since_last_use = 0;
      int gid = pool->pop[idx].global_id;
      if (gid >= 0 && gid < (int)pool->history.size()) {
        pool->history[gid].use_count++;
        pool->history[gid].rounds_since_last_use = 0;
        float sg = tree->split_gain[t];
        int p1 = pool->history[gid].parent1;
        int p2 = pool->history[gid].parent2;
        if (p1 >= 0) pool->propagate_credit(p1, sg, gamma, 1);
        if (p2 >= 0) pool->propagate_credit(p2, sg, gamma, 1);
      }
    }
  }

  // 5. Predict on full N using full_cache
  std::vector<int> node_assign(N, 0);
  int max_d = tree->max_depth;
  for (int depth = 0; depth < max_d; depth++) {
    int n_nodes = 1 << depth, base = n_nodes - 1;
    for (int local = 0; local < n_nodes; local++) {
      int t = base + local;
      if (tree->is_leaf[t]) continue;
      int p = tree->split_hyp_idx[t];
      float thr = tree->split_threshold[t];
      const float* fc = pool->pop[p].full_cache.data();
      int t_left = 2 * t + 1, t_right = 2 * t + 2;
      for (int i = 0; i < N; i++)
        if (node_assign[i] == t)
          node_assign[i] = (fc[i] < thr) ? t_left : t_right;
    }
  }
  for (int i = 0; i < N; i++) {
    const float* lv = tree->leaf_values.data() + (size_t)node_assign[i] * K;
    float* oi = out_pred + (size_t)i * K;
    for (int k = 0; k < K; k++) oi[k] = lv[k];
  }

  return static_cast<void*>(tree);
}

}  // extern "C"
