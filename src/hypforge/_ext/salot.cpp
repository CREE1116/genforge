// testforge.cpp — Residual-Driven Sparse Oblique Boosting (RSOB Engine v5.2 -
// PRODUCTION)
//
// Refined by Kri's Architectural Insights & Bug Fixes:
//   1. Solved Python Segfault Crash: Moved histogram vectors directly INSIDE
//   the parallel loop
//      to eliminate OpenMP private default-constructor empty vector allocation
//      bugs.
//   2. Restored 2D Oblique Boundaries: Implemented an adaptive dimension-guard
//   that disables
//      energy pruning for low-dimensional spaces (D <= 4) to prevent
//      axis-aligned collapse.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "bfstree_types.h"

static constexpr int TOP_M = 128;
static constexpr float EPS = 1e-8f;
static constexpr int D_SUB_MAX = 32;
static constexpr int SCORE_NS_MAX = 2000;
static constexpr int HIST_BINS = 64;
static constexpr float MIN_CHILD_W = 0.1f;

static inline float tapply_op(float val, int type) {
  return (type == 1) ? ((val > 0.0f) ? val : 0.01f * val) : val;
}

// ─── TForgeHyp ───────────────────────────────────────────────────────────────

struct TForgeHyp {
  int type = 0;
  std::vector<float> w;
  std::vector<int> nonzero_dims;

  int global_id = -1;
  int birth_round = 0;
  bool is_base = false;

  int use_count = 0;
  int rounds_since_last_use = 0;
  int n_scored = 0;

  float score = 0.0f;
  float fitness = 0.0f;

  std::vector<float> full_cache;
  std::vector<float> thresholds;
  float bin_min = 0.0f, bin_max = 1.0f;

  int h1_idx = -1, h2_idx = -1;
  int parent1 = -1, parent2 = -1;
  int family_id = -1;
  float family_fitness = 0.0f, breeding_value = 0.0f, ancestor_credit = 0.0f;
  int n_obs = 0;
  double mu_fitness = 0.0, M2_fitness = 0.0;
  int product_depth = 0;
  int stored_complexity = 0;

  int complexity() const { return stored_complexity; }

  void compute_nonzero_dims() {
    nonzero_dims.clear();
    for (int d = 0; d < (int)w.size(); d++)
      if (w[d] != 0.0f) nonzero_dims.push_back(d);
  }

  float project(const float* xi) const {
    float val = 0.0f;
    for (int d : nonzero_dims) val += w[d] * xi[d];
    return val;
  }
};

// ─── Utilities ───────────────────────────────────────────────────────────────

static std::vector<int> tf_random_indices(int N, int size, unsigned int seed) {
  std::vector<int> idxs(N);
  std::iota(idxs.begin(), idxs.end(), 0);
  if (N <= size) return idxs;
  std::mt19937 g(seed);
  for (int i = 0; i < size; i++) {
    int j = std::uniform_int_distribution<int>(i, N - 1)(g);
    std::swap(idxs[i], idxs[j]);
  }
  idxs.resize(size);
  return idxs;
}

static bool energy_prune_and_normalise(std::vector<float>& w,
                                       float energy_frac) {
  int n = (int)w.size();
  std::vector<int> ord(n);
  std::iota(ord.begin(), ord.end(), 0);
  std::sort(ord.begin(), ord.end(),
            [&](int a, int b) { return std::abs(w[a]) > std::abs(w[b]); });

  float total_sq = 0.0f;
  for (float v : w) total_sq += v * v;
  if (total_sq < 1e-10f) return false;

  float target = energy_frac * total_sq;
  float cum_sq = 0.0f;
  int keep = n;
  for (int i = 0; i < n; i++) {
    cum_sq += w[ord[i]] * w[ord[i]];
    if (cum_sq >= target) {
      keep = i + 1;
      break;
    }
  }

  for (int i = keep; i < n; i++) w[ord[i]] = 0.0f;

  float norm = 0.0f;
  for (float v : w) norm += v * v;
  norm = std::sqrt(norm);
  if (norm < 1e-5f) return false;
  for (float& v : w) v /= norm;
  return true;
}

static float gain_score(const TForgeHyp& h, const float* G_full,
                        const float* H_full, const int* sub, int Ns, int K,
                        float reg_lambda) {
  if (h.full_cache.empty()) return 0.0f;

  float min_v = 1e30f, max_v = -1e30f;
  for (int j = 0; j < Ns; j++) {
    float v = h.full_cache[sub[j]];
    if (v < min_v) min_v = v;
    if (v > max_v) max_v = v;
  }
  if (max_v - min_v < 1e-5f) return 0.0f;

  std::vector<float> bin_G((size_t)HIST_BINS * K, 0.0f);
  std::vector<float> bin_H((size_t)HIST_BINS * K, 0.0f);
  float scale = (float)HIST_BINS / (max_v - min_v + EPS);

  for (int j = 0; j < Ns; j++) {
    int idx = sub[j];
    int b = (int)((h.full_cache[idx] - min_v) * scale);
    if (b < 0) b = 0;
    if (b >= HIST_BINS) b = HIST_BINS - 1;

    size_t b_offset = (size_t)b * K;
    const float* gj = G_full + (size_t)idx * K;
    const float* hj = H_full + (size_t)idx * K;
    for (int c = 0; c < K; c++) {
      bin_G[b_offset + c] += gj[c];
      bin_H[b_offset + c] += hj[c];
    }
  }

  std::vector<float> Gt(K, 0.0f), Ht(K, 0.0f);
  for (int b = 0; b < HIST_BINS; b++) {
    size_t b_offset = (size_t)b * K;
    for (int c = 0; c < K; c++) {
      Gt[c] += bin_G[b_offset + c];
      Ht[c] += bin_H[b_offset + c];
    }
  }

  float total_gain = 0.0f;
  for (int c = 0; c < K; c++) {
    float total_base = -0.5f * Gt[c] * Gt[c] / (Ht[c] + reg_lambda + EPS);
    float Gc = 0.0f, Hc = 0.0f, max_pk = 0.0f;
    for (int b = 0; b < HIST_BINS - 1; b++) {
      size_t b_offset = (size_t)b * K;
      Gc += bin_G[b_offset + c];
      Hc += bin_H[b_offset + c];
      float Gr = Gt[c] - Gc, Hr = Ht[c] - Hc;
      if (Hc < MIN_CHILD_W || Hr < MIN_CHILD_W) continue;

      float v = 0.5f * (Gc * Gc / (Hc + reg_lambda + EPS) +
                        Gr * Gr / (Hr + reg_lambda + EPS)) +
                total_base;
      if (v > max_pk) max_pk = v;
    }
    total_gain += max_pk;
  }

  return total_gain / (float)K;
}

static std::vector<float> cholesky_solve(const std::vector<float>& A,
                                         const std::vector<float>& r, int n) {
  std::vector<float> L(n * n, 0.0f);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++) {
      float sum = 0.0f;
      for (int k = 0; k < j; k++) sum += L[i * n + k] * L[j * n + k];
      if (i == j) {
        float val = A[i * n + i] - sum;
        if (val < 1e-6f) val = 1e-6f;
        L[i * n + j] = std::sqrt(val);
      } else {
        L[i * n + j] = (A[i * n + j] - sum) / L[j * n + j];
      }
    }
  }
  std::vector<float> y(n, 0.0f);
  for (int i = 0; i < n; i++) {
    float sum = 0.0f;
    for (int k = 0; k < i; k++) sum += L[i * n + k] * y[k];
    y[i] = (r[i] - sum) / L[i * n + i];
  }
  std::vector<float> x(n, 0.0f);
  for (int i = n - 1; i >= 0; i--) {
    float sum = 0.0f;
    for (int k = i + 1; k < n; k++) sum += L[k * n + i] * x[k];
    x[i] = (y[i] - sum) / L[i * n + i];
  }
  return x;
}

static std::vector<float> wls_block_solve(const std::vector<int>& block, int D,
                                          const float* X, const float* G,
                                          const float* H, const int* sub,
                                          int Ns, int K, int k_class,
                                          float reg_lambda, float energy_frac) {
  int b = (int)block.size();
  std::vector<float> A(b * b, 0.0f);
  std::vector<float> r(b, 0.0f);

  for (int j = 0; j < Ns; j++) {
    int idx = sub[j];
    float hk = H[idx * K + k_class];
    float gk = G[idx * K + k_class];
    for (int i = 0; i < b; i++) {
      float xi = X[idx * D + block[i]];
      r[i] -= xi * gk;
      for (int l = 0; l <= i; l++) {
        float val = hk * xi * X[idx * D + block[l]];
        A[i * b + l] += val;
        if (i != l) A[l * b + i] += val;
      }
    }
  }
  for (int i = 0; i < b; i++) A[i * b + i] += reg_lambda;

  std::vector<float> w_b = cholesky_solve(A, r, b);

  if (!energy_prune_and_normalise(w_b, energy_frac)) return {};

  std::vector<float> w(D, 0.0f);
  for (int i = 0; i < b; i++) w[block[i]] = w_b[i];
  return w;
}

// ─── TestForgePool
// ────────────────────────────────────────────────────────────

class TestForgePool {
 public:
  int D, max_size, N = 0;
  int op_mode = 0;
  int crossover_top_k = 6;
  float energy_frac = 0.75f;
  unsigned int seed = 42;

  std::vector<TForgeHyp> pop;
  std::vector<TForgeHyp> history;
  float cx_births[3][3] = {};
  float cx_survivors[3][3] = {};

  TestForgePool(int D_, int max_size_) : D(D_), max_size(max_size_) {
    for (int j = 0; j < D_; j++) {
      TForgeHyp h;
      h.type = 0;
      h.w.assign(D_, 0.0f);
      h.w[j] = 1.0f;
      h.nonzero_dims = {j};
      h.is_base = true;
      h.global_id = j;
      h.family_id = j;
      h.stored_complexity = 1;
      TForgeHyp hh = h;
      hh.full_cache.clear();
      hh.thresholds.clear();
      history.push_back(std::move(hh));
      pop.push_back(std::move(h));
    }
  }

  void build_cache_and_thresholds(TForgeHyp& h, const float* X,
                                  unsigned int lseed) {
    if ((int)h.full_cache.size() != N) {
      if (h.nonzero_dims.empty()) h.compute_nonzero_dims();
      h.full_cache.resize(N);
      for (int i = 0; i < N; i++)
        h.full_cache[i] = tapply_op(h.project(X + (size_t)i * D), h.type);
    }
    if (!h.thresholds.empty()) return;
    auto ri = tf_random_indices(N, std::min(N, 10000), lseed);
    int Ns_q = (int)ri.size();
    std::vector<float> samp(Ns_q);
    for (int i = 0; i < Ns_q; i++) samp[i] = h.full_cache[ri[i]];
    std::sort(samp.begin(), samp.end());
    h.thresholds.resize(9);
    for (int q = 0; q < 9; q++) {
      int idx = (int)((q + 1) * 0.1f * Ns_q);
      h.thresholds[q] = samp[std::min(idx, Ns_q - 1)];
    }
    h.bin_min = samp.front();
    h.bin_max = samp.back();
  }

  void ensure_caches(const float* X) {
    int P = (int)pop.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int p = 0; p < P; p++) {
      auto& h = pop[p];
      if (h.w.empty()) continue;
      if (h.nonzero_dims.empty()) h.compute_nonzero_dims();
      build_cache_and_thresholds(h, X, seed + (unsigned)p);
    }
  }

  void eval(const float* X, int N_in, float* out_Z) const {
    int P = (int)pop.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int p = 0; p < P; p++) {
      const auto& h = pop[p];
      float* out = out_Z + (size_t)p * N_in;
      for (int i = 0; i < N_in; i++)
        out[i] = tapply_op(h.project(X + (size_t)i * D), h.type);
    }
  }

  void evolve(const float* X, const float* G_full, const float* H_full,
              const int* sub_indices, int N_in, int Ns, int K, int D_num,
              float reg_lambda, float /*eta_penalty*/ = 0.0f,
              int current_round = 0) {
    this->N = N_in;
    seed += (unsigned)current_round + 1;

    ensure_caches(X);

    std::mt19937 rng(seed ^ 0xC0FFEEu);
    int Ns_score = std::min(Ns, SCORE_NS_MAX);

    int d_sub = std::max(2, std::min(D_SUB_MAX, (int)std::sqrt((float)D_num)));
    int n_blocks = (D_num + d_sub - 1) / d_sub;

    std::vector<int> perm(D_num);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);

    std::vector<std::vector<int>> blocks(n_blocks);
    for (int bi = 0; bi < n_blocks; bi++) {
      int start = bi * d_sub, end = std::min(start + d_sub, D_num);
      blocks[bi].assign(perm.begin() + start, perm.begin() + end);
    }

    int n_cands = K + n_blocks * K;
    std::vector<std::vector<float>> cand_ws(n_cands);
    std::vector<bool> cand_ok(n_cands, false);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int task = 0; task < n_cands; task++) {
      if (task < K) {
        int c = task;
        std::vector<float> v(D, 0.0f);
        for (int d = 0; d < D_num; d++) {
          float num = 0.0f, den = 0.0f;
          for (int j = 0; j < Ns; j++) {
            int idx = sub_indices[j];
            float xd = X[idx * D + d];
            num -= xd * G_full[idx * K + c];
            den += H_full[idx * K + c] * xd * xd;
          }
          v[d] = num / (den + reg_lambda + EPS);
        }
        if (energy_prune_and_normalise(v, energy_frac)) {
          cand_ws[task] = std::move(v);
          cand_ok[task] = true;
        }
      } else {
        int bi = (task - K) / K;
        int c = (task - K) % K;
        std::vector<float> w =
            wls_block_solve(blocks[bi], D, X, G_full, H_full, sub_indices, Ns,
                            K, c, reg_lambda, energy_frac);
        if (!w.empty()) {
          cand_ws[task] = std::move(w);
          cand_ok[task] = true;
        }
      }
    }

    int added = 0;
    for (int ci = 0; ci < n_cands; ci++) {
      if (!cand_ok[ci]) continue;

      TForgeHyp h;
      h.type = 0;
      h.w = std::move(cand_ws[ci]);
      h.birth_round = current_round;
      h.is_base = false;
      h.global_id = (int)history.size();
      h.family_id = h.global_id;
      h.compute_nonzero_dims();
      h.stored_complexity = (int)h.nonzero_dims.size();

      h.full_cache.resize(N_in);
      for (int i = 0; i < N_in; i++)
        h.full_cache[i] = h.project(X + (size_t)i * D);

      TForgeHyp hc = h;
      hc.full_cache.clear();
      hc.full_cache.shrink_to_fit();
      history.push_back(std::move(hc));

      pop.push_back(std::move(h));
      added++;
    }

    int P = (int)pop.size();
    int P_before = P - added;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int p = P_before; p < P; p++) {
      auto& h = pop[p];
      h.score =
          gain_score(h, G_full, H_full, sub_indices, Ns_score, K, reg_lambda);
      h.fitness = h.score;
      h.n_scored++;
    }

    // Base axes always kept; non-base sorted by score, capped at max_size.
    std::stable_sort(pop.begin(), pop.end(),
                     [](const TForgeHyp& a, const TForgeHyp& b) {
                       if (a.is_base != b.is_base)
                         return (int)a.is_base > (int)b.is_base;
                       return a.score > b.score;
                     });
    if ((int)pop.size() > max_size) pop.resize(max_size);

    for (auto& h : pop) {
      h.rounds_since_last_use++;
      int gid = h.global_id;
      if (gid >= 0 && gid < (int)history.size())
        history[gid].score = history[gid].fitness = h.score;
    }
  }

  // ── SALOT: Shared-Axis Leaf-wise Oblique Tree (RSOB Engine 마스터 코어)
  // ──────
  BFSTree* salot_build_tree(const float* X, const float* G, const float* H,
                            const int* tree_sub, int Ns, int N_in, int K,
                            int D_num, int max_depth, float reg_lambda,
                            int current_round) {
    int D = this->D;
    int max_nodes = (1 << (max_depth + 1)) - 1;

    auto* tree = new BFSTree();
    tree->K = K;
    tree->max_depth = max_depth;
    tree->total_nodes = max_nodes;
    tree->is_leaf.assign(max_nodes, 1);
    tree->split_hyp_idx.assign(max_nodes, -1);
    tree->split_threshold.assign(max_nodes, 0.0f);
    tree->leaf_values.assign((size_t)max_nodes * K, 0.0f);
    tree->split_gain.assign(max_nodes, 0.0f);

    std::vector<std::vector<int>> node_samp(max_nodes);
    node_samp[0].assign(tree_sub, tree_sub + Ns);

    for (int b = 0; b < std::min(D, (int)pop.size()); b++) {
      auto& hb = pop[b];
      if ((int)hb.full_cache.size() != N_in) {
        hb.full_cache.resize(N_in);
        for (int i = 0; i < N_in; i++)
          hb.full_cache[i] = hb.project(X + (size_t)i * D);
      }
    }

    for (int d = 0; d < max_depth; d++) {
      int first_node = (1 << d) - 1;
      int n_at_depth = 1 << d;

      std::vector<int> active_nodes;
      std::vector<int> layer_samp;
      for (int local = 0; local < n_at_depth; local++) {
        int t = first_node + local;
        if ((int)node_samp[t].size() < 20) continue;
        active_nodes.push_back(t);
        layer_samp.insert(layer_samp.end(), node_samp[t].begin(),
                          node_samp[t].end());
      }
      if (active_nodes.empty()) break;

      // ── Step 2–3: Per-class diagonal WLS (parallel over classes) ────────
      // Σ_k G[i,k] = 0 (softmax) → all-class sum always cancels to zero.
      // Compute K separate axes and pick max-norm. OpenMP over K classes.
      std::vector<int> p_layer_axes;
      {
        int nl = (int)layer_samp.size();
        std::vector<std::vector<float>> all_w(K, std::vector<float>(D, 0.0f));
        std::vector<float> all_sq(K, 0.0f);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int c = 0; c < K; c++) {
          std::vector<float> mean_xj(D_num, 0.0f);
          for (int ji = 0; ji < nl; ji++) {
            int j = layer_samp[ji];
            const float* xj = X + (size_t)j * D;
            for (int f = 0; f < D_num; f++) mean_xj[f] += xj[f];
          }
          for (int f = 0; f < D_num; f++) mean_xj[f] /= (float)nl;

          std::vector<float> num_c(D_num, 0.0f), den_c(D_num, 0.0f);
          for (int ji = 0; ji < nl; ji++) {
            int j = layer_samp[ji];
            const float* xj = X + (size_t)j * D;
            float gc = G[(size_t)j * K + c];
            float hc = H[(size_t)j * K + c];
            for (int f = 0; f < D_num; f++) {
              float x_centered = xj[f] - mean_xj[f];
              num_c[f] -= x_centered * gc;
              den_c[f] += hc * x_centered * x_centered;
            }
          }
          float sq = 0.0f;
          for (int f = 0; f < D_num; f++) {
            float w = num_c[f] / (den_c[f] + reg_lambda + EPS);
            all_w[c][f] = w;
            sq += w * w;
          }
          all_sq[c] = sq;
        }

        for (int c = 0; c < K; c++) {
          std::vector<float> w_c = all_w[c];
          if (!energy_prune_and_normalise(w_c, energy_frac)) continue;
          TForgeHyp h_ax;
          h_ax.type = 0;
          h_ax.w = std::move(w_c);
          h_ax.is_base = false;
          h_ax.global_id = (int)history.size();
          h_ax.family_id = h_ax.global_id;
          h_ax.birth_round = current_round;
          h_ax.compute_nonzero_dims();
          h_ax.stored_complexity = (int)h_ax.nonzero_dims.size();
          h_ax.full_cache.resize(N_in);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
          for (int i = 0; i < N_in; i++)
            h_ax.full_cache[i] = h_ax.project(X + (size_t)i * D);
          history.push_back(h_ax);
          p_layer_axes.push_back((int)pop.size());
          pop.push_back(std::move(h_ax));
        }
      }
      if (p_layer_axes.empty()) continue;  // no valid WLS axis at this depth

      // ── Step 5: Multi-Axis Histogram Swarm Scan ──
      int n_active = (int)active_nodes.size();

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
      for (int a_idx = 0; a_idx < n_active; a_idx++) {
        int t_node = active_nodes[a_idx];
        auto& samp = node_samp[t_node];
        int ns = (int)samp.size();

        int tl = 2 * t_node + 1, tr_node = 2 * t_node + 2;

        std::vector<float> Gt(K, 0.0f), Ht(K, 0.0f);
        for (int j : samp) {
          const float* gj = G + (size_t)j * K;
          const float* hj = H + (size_t)j * K;
          for (int c = 0; c < K; c++) {
            Gt[c] += gj[c];
            Ht[c] += hj[c];
          }
        }

        float total_base = 0.0f;
        for (int c = 0; c < K; c++)
          total_base -= 0.5f * Gt[c] * Gt[c] / (Ht[c] + reg_lambda + EPS);

        float best_gain = 0.0f, best_thr = 0.0f;
        int best_axis_idx = -1;
        bool found = false;

        // [FIXED] OpenMP 메모리 격리를 위해 스레드 워커 내부 공간에 로컬 백
        // 생성 (Segfault 완벽 박멸)
        std::vector<float> thread_bin_G((size_t)HIST_BINS * K, 0.0f);
        std::vector<float> thread_bin_H((size_t)HIST_BINS * K, 0.0f);
        std::vector<int> thread_bin_cnt(HIST_BINS, 0);

        for (int p_idx : p_layer_axes) {
          const float* fc = pop[p_idx].full_cache.data();

          float min_v = 1e30f, max_v = -1e30f;
          for (int j : samp) {
            float v = fc[j];
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
          }
          if (max_v - min_v < 1e-5f) continue;

          std::fill(thread_bin_G.begin(), thread_bin_G.end(), 0.0f);
          std::fill(thread_bin_H.begin(), thread_bin_H.end(), 0.0f);
          std::fill(thread_bin_cnt.begin(), thread_bin_cnt.end(), 0);

          float scale = (float)HIST_BINS / (max_v - min_v + EPS);
          for (int j : samp) {
            int b = (int)((fc[j] - min_v) * scale);
            if (b < 0) b = 0;
            if (b >= HIST_BINS) b = HIST_BINS - 1;

            thread_bin_cnt[b]++;
            size_t b_offset = (size_t)b * K;
            const float* gj = G + (size_t)j * K;
            const float* hj = H + (size_t)j * K;
            for (int c = 0; c < K; c++) {
              thread_bin_G[b_offset + c] += gj[c];
              thread_bin_H[b_offset + c] += hj[c];
            }
          }

          std::vector<float> Gc(K, 0.0f), Hc(K, 0.0f);
          int cum_left_cnt = 0;

          for (int b = 0; b < HIST_BINS - 1; b++) {
            cum_left_cnt += thread_bin_cnt[b];
            size_t b_offset = (size_t)b * K;
            for (int c = 0; c < K; c++) {
              Gc[c] += thread_bin_G[b_offset + c];
              Hc[c] += thread_bin_H[b_offset + c];
            }
            int cum_right_cnt = (int)samp.size() - cum_left_cnt;
            if (cum_left_cnt < 10 || cum_right_cnt < 10) continue;

            // Check total hessian across classes (not per-class).
            // Per-class check was too strict for imbalanced multiclass: one
            // rare class failing MIN_CHILD_W would reject otherwise valid
            // splits.
            float Hc_sum = 0.0f, Hr_sum = 0.0f;
            for (int c = 0; c < K; c++) {
              Hc_sum += Hc[c];
              Hr_sum += Ht[c] - Hc[c];
            }
            if (Hc_sum < MIN_CHILD_W || Hr_sum < MIN_CHILD_W) continue;

            float gain = total_base;
            for (int c = 0; c < K; c++) {
              float Gr = Gt[c] - Gc[c], Hr = Ht[c] - Hc[c];
              gain += 0.5f * (Gc[c] * Gc[c] / (Hc[c] + reg_lambda + EPS) +
                              Gr * Gr / (Hr + reg_lambda + EPS));
            }

            if (gain > best_gain) {
              best_gain = gain;
              best_thr =
                  min_v + ((float)(b + 1) / (float)HIST_BINS) * (max_v - min_v);
              best_axis_idx = p_idx;
              found = true;
            }
          }
        }

        if (found) {
          tree->is_leaf[t_node] = 0;
          tree->split_hyp_idx[t_node] = best_axis_idx;
          tree->split_threshold[t_node] = best_thr;
          tree->split_gain[t_node] = best_gain;

          const float* best_fc = pop[best_axis_idx].full_cache.data();
          std::vector<int> left_sub, right_sub;
          for (int j : samp) {
            if (best_fc[j] < best_thr)
              left_sub.push_back(j);
            else
              right_sub.push_back(j);
          }
          node_samp[tl] = std::move(left_sub);
          node_samp[tr_node] = std::move(right_sub);
        }
      }
    }

    for (int t = 0; t < max_nodes; t++) {
      if (!tree->is_leaf[t]) continue;
      if (node_samp[t].empty()) continue;
      for (int c = 0; c < K; c++) {
        float Gs = 0.0f, Hs = 0.0f;
        for (int j : node_samp[t]) {
          Gs += G[(size_t)j * K + c];
          Hs += H[(size_t)j * K + c];
        }
        tree->leaf_values[(size_t)t * K + c] = -Gs / (Hs + reg_lambda + EPS);
      }
    }
    return tree;
  }
};

// ─── C API ───────────────────────────────────────────────────────────────────

extern "C" {

void* tpool_create(int D, int max_size, int) {
  return static_cast<void*>(new TestForgePool(D, max_size));
}
void tpool_free(void* h) { delete static_cast<TestForgePool*>(h); }

void tpool_set_options(void* handle, int op_mode, int crossover_top_k, int, int,
                       int, int, int, float energy_frac, float) {
  auto* p = static_cast<TestForgePool*>(handle);
  p->op_mode = op_mode;
  p->crossover_top_k = crossover_top_k;
  if (energy_frac > 0.0f) p->energy_frac = energy_frac;
}

void tpool_evolve(void* handle, const float* X, const float* G, const float* H,
                  const int* sub, int N, int Ns, int K, int D_num,
                  float reg_lambda, float eta, int round) {
  static_cast<TestForgePool*>(handle)->evolve(X, G, H, sub, N, Ns, K, D_num,
                                              reg_lambda, eta, round);
}

void tpool_eval(void* handle, const float* X, int N, float* out) {
  static_cast<const TestForgePool*>(handle)->eval(X, N, out);
}

int tpool_get_size(void* handle) {
  return (int)static_cast<const TestForgePool*>(handle)->pop.size();
}

int tpool_get_history_size(void* handle) {
  return (int)static_cast<const TestForgePool*>(handle)->history.size();
}

void tpool_get_active_indices(void* handle, int* out) {
  auto* pool = static_cast<TestForgePool*>(handle);
  for (int p = 0; p < (int)pool->pop.size(); p++)
    out[p] = pool->pop[p].global_id;
}

void tpool_get_caches_and_thresholds(void* handle, float* out_Z,
                                     float* out_thr) {
  auto* pool = static_cast<TestForgePool*>(handle);
  int P = (int)pool->pop.size(), N = pool->N;
  for (int p = 0; p < P; p++) {
    std::memcpy(out_Z + (size_t)p * N, pool->pop[p].full_cache.data(),
                N * sizeof(float));
    for (int q = 0; q < 9; q++)
      out_thr[q * P + p] = (!pool->pop[p].thresholds.empty())
                               ? pool->pop[p].thresholds[q]
                               : 0.0f;
  }
}

void tpool_update_use_counts(void* handle, const int* split_indices,
                             int n_nodes) {
  auto* pool = static_cast<TestForgePool*>(handle);
  int P = (int)pool->pop.size();
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

void tpool_export(void* handle, int* types, float* weights, int* h1i, int* h2i,
                  int* uc, int* rslu, float* credits, uint8_t* is_base, int* p1,
                  int* p2, int* br, int* fid) {
  auto* pool = static_cast<TestForgePool*>(handle);
  int H = (int)pool->history.size(), D = pool->D;
  for (int p = 0; p < H; p++) {
    const auto& h = pool->history[p];
    types[p] = h.type;
    h1i[p] = h.h1_idx;
    h2i[p] = h.h2_idx;
    uc[p] = h.use_count;
    rslu[p] = h.rounds_since_last_use;
    credits[p] = h.score;
    is_base[p] = h.is_base ? 1 : 0;
    p1[p] = h.parent1;
    p2[p] = h.parent2;
    br[p] = h.birth_round;
    fid[p] = h.family_id;
    if (!h.w.empty())
      std::memcpy(weights + p * D, h.w.data(), D * sizeof(float));
    else
      std::memset(weights + p * D, 0, D * sizeof(float));
  }
}

void* tpool_import(int D, int max_size, int U, const int* types,
                   const float* weights, const int* h1i, const int* h2i,
                   const int* uc, const int* rslu, const float* credits,
                   const uint8_t* is_base, const int* p1, const int* p2,
                   const int* br, const int* fid, int P, const int* active) {
  auto* pool = new TestForgePool(D, max_size);
  pool->pop.clear();
  pool->history.clear();
  for (int u = 0; u < U; u++) {
    TForgeHyp h;
    h.type = types[u];
    h.h1_idx = h1i[u];
    h.h2_idx = h2i[u];
    h.use_count = uc[u];
    h.rounds_since_last_use = rslu[u];
    h.score = h.fitness = credits[u];
    h.is_base = is_base[u] != 0;
    h.parent1 = p1[u];
    h.parent2 = p2[u];
    h.birth_round = br[u];
    h.family_id = fid[u];
    h.global_id = u;
    if (h.type < 2) {
      h.w.assign(weights + u * D, weights + (u + 1) * D);
      h.compute_nonzero_dims();
    }
    pool->history.push_back(std::move(h));
  }
  for (int pp = 0; pp < P; pp++) {
    int idx = active[pp];
    if (idx >= 0 && idx < U) pool->pop.push_back(pool->history[idx]);
  }
  return static_cast<void*>(pool);
}

void tpool_get_transition_matrix(void* handle, float* births,
                                 float* survivors) {
  auto* pool = static_cast<TestForgePool*>(handle);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++) {
      births[i * 3 + j] = pool->cx_births[i][j];
      survivors[i * 3 + j] = pool->cx_survivors[i][j];
    }
}

void* tpool_build_tree(void* pool_handle, const float* X, const float* G_full,
                       const float* H_full, int N, int K, const int* evolve_sub,
                       int Ns_evolve, const int* tree_sub, int Ns_tree,
                       int D_num, int max_depth, float reg_lambda,
                       float eta_penalty, int do_evolve, float* out_pred,
                       int current_round) {
  auto* pool = static_cast<TestForgePool*>(pool_handle);
  if (do_evolve)
    pool->evolve(X, G_full, H_full, evolve_sub, N, Ns_evolve, K, D_num,
                 reg_lambda, eta_penalty, current_round);

  int P = (int)pool->pop.size();
  if (P == 0) return nullptr;

  int P_use = std::min(P, TOP_M);
  std::vector<const float*> caches(P_use);
  for (int p = 0; p < P_use; p++) caches[p] = pool->pop[p].full_cache.data();
  std::vector<int> tree_indices(tree_sub, tree_sub + Ns_tree);

  BFSTree* tree =
      bfs_build_best_first(caches.data(), P_use, tree_indices, G_full, H_full,
                           K, max_depth, 20, 10, reg_lambda);

  for (int p = 0; p < P; p++) {
    pool->pop[p].rounds_since_last_use++;
    int gid = pool->pop[p].global_id;
    if (gid >= 0 && gid < (int)pool->history.size())
      pool->history[gid].rounds_since_last_use++;
  }
  if (tree) {
    for (int t = 0; t < tree->total_nodes; t++) {
      int idx = tree->split_hyp_idx[t];
      if (idx >= 0 && idx < P_use) {
        pool->pop[idx].use_count++;
        pool->pop[idx].rounds_since_last_use = 0;
        int gid = pool->pop[idx].global_id;
        if (gid >= 0 && gid < (int)pool->history.size()) {
          pool->history[gid].use_count++;
          pool->history[gid].rounds_since_last_use = 0;
        }
      }
    }
    std::vector<int> node_assign(N, 0);
    for (int depth = 0; depth < tree->max_depth; depth++) {
      int n_nodes = 1 << depth, base = n_nodes - 1;
      for (int local = 0; local < n_nodes; local++) {
        int t = base + local;
        if (tree->is_leaf[t]) continue;
        int p_idx = tree->split_hyp_idx[t];
        float thr = tree->split_threshold[t];
        const float* fc = pool->pop[p_idx].full_cache.data();
        int tl = 2 * t + 1, tr = 2 * t + 2;
        for (int i = 0; i < N; i++)
          if (node_assign[i] == t) node_assign[i] = (fc[i] < thr) ? tl : tr;
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

void* tpool_build_salot_tree(void* pool_handle, const float* X,
                             const float* G_full, const float* H_full, int N,
                             int K, const int* evolve_sub, int Ns_evolve,
                             const int* tree_sub, int Ns_tree, int D_num,
                             int max_depth, float reg_lambda, float eta_penalty,
                             int do_evolve, float* out_pred,
                             int current_round) {
  auto* pool = static_cast<TestForgePool*>(pool_handle);
  pool->N = N;

  if (do_evolve)
    pool->evolve(X, G_full, H_full, evolve_sub, N, Ns_evolve, K, D_num,
                 reg_lambda, eta_penalty, current_round);

  BFSTree* tree =
      pool->salot_build_tree(X, G_full, H_full, tree_sub, Ns_tree, N, K, D_num,
                             max_depth, reg_lambda, current_round);

  if (tree) {
    for (int t = 0; t < tree->total_nodes; t++) {
      if (tree->is_leaf[t]) continue;
      int idx = tree->split_hyp_idx[t];
      if (idx >= 0 && idx < (int)pool->pop.size()) {
        pool->pop[idx].use_count++;
        pool->pop[idx].rounds_since_last_use = 0;
        int gid = pool->pop[idx].global_id;
        if (gid >= 0 && gid < (int)pool->history.size()) {
          pool->history[gid].use_count++;
          pool->history[gid].rounds_since_last_use = 0;
        }
      }
    }
    std::vector<int> node_assign(N, 0);
    for (int depth = 0; depth < tree->max_depth; depth++) {
      int n_nodes = 1 << depth, base = n_nodes - 1;
      for (int local = 0; local < n_nodes; local++) {
        int t = base + local;
        if (tree->is_leaf[t]) continue;
        int p_idx = tree->split_hyp_idx[t];
        float thr = tree->split_threshold[t];
        const float* fc = pool->pop[p_idx].full_cache.data();
        int tl = 2 * t + 1, tr = 2 * t + 2;
        for (int i = 0; i < N; i++)
          if (node_assign[i] == t) node_assign[i] = (fc[i] < thr) ? tl : tr;
      }
    }
    for (int i = 0; i < N; i++) {
      const float* lv = tree->leaf_values.data() + (size_t)node_assign[i] * K;
      float* oi = out_pred + (size_t)i * K;
      for (int k = 0; k < K; k++) oi[k] = lv[k];
    }
  }

  for (int p = 0; p < (int)pool->pop.size(); p++)
    pool->pop[p].rounds_since_last_use++;

  return static_cast<void*>(tree);
}

// ── Pool-less Local Stochastic Oblique Boosting ───────────────────────────────
// Per-node: block WLS (true oblique) + instance subsampling + feature
// subsampling + zero-cross-term bundling.  No persistent pool, no global cache.

static void _salot_route(const BFSTree* tree, const float* X, int N,
                         float* out_pred) {
  int D = tree->D, K = tree->K;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++) {
    const float* xi = X + (size_t)i * D;
    int t = 0;
    for (int dep = 0; dep < tree->max_depth; dep++) {
      if (tree->is_leaf[t]) break;
      const float* tw = tree->split_weights.data() + (size_t)t * D;
      float proj = 0.0f;
      for (int f = 0; f < D; f++) proj += tw[f] * xi[f];
      t = (proj < tree->split_threshold[t]) ? (2 * t + 1) : (2 * t + 2);
    }
    const float* lv = tree->leaf_values.data() + (size_t)t * K;
    float* oi = out_pred + (size_t)i * K;
    for (int k = 0; k < K; k++) oi[k] = lv[k];
  }
}

// Pre-compute global exclusive feature-pair mask.
// Only runs when D_num ≤ 150 (O(D²) cost); for high-D we use local bundling
// inside node_block_wls that checks only the d_sub sampled features.
static std::vector<bool> compute_bundle_matrix(const float* X, int D, int D_num,
                                               const int* sub, int Ns) {
  std::vector<bool> excl;
  if (D_num > 150) return excl;

  int ns = std::min(Ns, 2000);
  std::vector<int> cooccur((size_t)D_num * D_num, 0);
  for (int si = 0; si < ns; si++) {
    const float* xi = X + (size_t)sub[si] * D;
    for (int j = 0; j < D_num; j++) {
      if (xi[j] == 0.0f) continue;
      for (int k = j + 1; k < D_num; k++) {
        if (xi[k] != 0.0f) cooccur[(size_t)j * D_num + k]++;
      }
    }
  }
  excl.assign((size_t)D_num * D_num, false);
  constexpr float BUNDLE_EPS = 0.05f;
  for (int j = 0; j < D_num; j++)
    for (int k = j + 1; k < D_num; k++) {
      float rate = (float)cooccur[(size_t)j * D_num + k] / ns;
      if (rate < BUNDLE_EPS)
        excl[(size_t)j * D_num + k] = excl[(size_t)k * D_num + j] = true;
    }
  return excl;
}

// Local bundling for high-D: check co-occurrence only among d_sub sampled
// features using node's own WLS subsample.  Returns d×d exclusivity matrix.
static std::vector<bool> local_bundle_check(
    const std::vector<int>& wls_samp, const std::vector<int>& feat_sub,
    int D, const float* X) {
  int d  = (int)feat_sub.size();
  int ns = (int)wls_samp.size();
  std::vector<int> cooccur(d * d, 0);
  for (int i = 0; i < ns; i++) {
    const float* xi = X + (size_t)wls_samp[i] * D;
    for (int fi = 0; fi < d; fi++) {
      if (xi[feat_sub[fi]] == 0.0f) continue;
      for (int fj = fi + 1; fj < d; fj++) {
        if (xi[feat_sub[fj]] != 0.0f) cooccur[fi * d + fj]++;
      }
    }
  }
  std::vector<bool> excl_local(d * d, false);
  constexpr float BUNDLE_EPS = 0.05f;
  for (int fi = 0; fi < d; fi++)
    for (int fj = fi + 1; fj < d; fj++) {
      float rate = (float)cooccur[fi * d + fj] / ns;
      if (rate < BUNDLE_EPS)
        excl_local[fi * d + fj] = excl_local[fj * d + fi] = true;
    }
  return excl_local;
}

// Per-node block WLS on a single representative class.
// Instead of K independent Cholesky solves, pick the class with the highest
// total |gradient| mass — O(K) scan cost, then one O(d³) solve.
//
// For global excl (D_num ≤ 150): index as excl[fi_real * D_num + fj_real].
// For high-D: pass empty excl + non-empty excl_local (d×d, local indices).
static std::vector<float> node_block_wls(
    const std::vector<int>& wls_samp, const std::vector<int>& feat_sub,
    int D, int D_num, const float* X, const float* G, const float* H, int K,
    float reg_lambda, float energy_frac,
    const std::vector<bool>& excl,        // global D_num×D_num, may be empty
    const std::vector<bool>& excl_local)  // local d×d, may be empty
{
  int nw = (int)wls_samp.size();
  int d  = (int)feat_sub.size();

  // Pick representative class: highest sum of |g_i|
  int best_c = 0;
  float best_g_mass = 0.0f;
  for (int c = 0; c < K; c++) {
    float mass = 0.0f;
    for (int i : wls_samp) mass += std::abs(G[(size_t)i * K + c]);
    if (mass > best_g_mass) { best_g_mass = mass; best_c = c; }
  }

  std::vector<float> A(d * d, 0.0f);
  std::vector<float> r(d, 0.0f);

  for (int i = 0; i < nw; i++) {
    int idx = wls_samp[i];
    float hk = H[(size_t)idx * K + best_c];
    float gk = G[(size_t)idx * K + best_c];
    const float* xi = X + (size_t)idx * D;
    for (int fi = 0; fi < d; fi++) {
      float xfi = xi[feat_sub[fi]];
      r[fi] -= xfi * gk;
      A[fi * d + fi] += hk * xfi * xfi;
      for (int fj = fi + 1; fj < d; fj++) {
        // Zero-cross-term bundling: skip exclusive pairs
        bool is_excl = false;
        if (!excl_local.empty()) {
          is_excl = excl_local[(size_t)fi * d + fj];
        } else if (!excl.empty()) {
          int fi_r = feat_sub[fi], fj_r = feat_sub[fj];
          is_excl = excl[(size_t)fi_r * D_num + fj_r];
        }
        if (is_excl) continue;
        float val = hk * xfi * xi[feat_sub[fj]];
        A[fi * d + fj] += val;
        A[fj * d + fi] += val;
      }
    }
  }
  for (int fi = 0; fi < d; fi++) A[fi * d + fi] += reg_lambda;

  std::vector<float> w_d = cholesky_solve(A, r, d);

  std::vector<float> w(D, 0.0f);
  for (int fi = 0; fi < d; fi++) w[feat_sub[fi]] = w_d[fi];

  std::vector<float> w_num(D_num);
  for (int f = 0; f < D_num; f++) w_num[f] = w[f];
  if (!energy_prune_and_normalise(w_num, energy_frac)) return {};
  for (int f = 0; f < D_num; f++) w[f] = w_num[f];
  return w;
}

// salot_build — Pool-less Local Stochastic Oblique GBDT tree builder
//
// Parameters added vs old version:
//   n_wls_max   : max instance subsample for WLS per node (e.g. 512)
//   d_sub_max   : max feature dimension for WLS block (e.g. 32)
//   energy_frac : energy-pruning threshold (e.g. 0.90)
//   seed        : random seed for per-node subsampling
void* salot_build(const float* X, int N, int D, int D_num, const float* G,
                  const float* H, int K, const int* sub, int Ns, int max_depth,
                  float reg_lambda, int n_wls_max, int d_sub_max,
                  float energy_frac, unsigned int seed, float* out_pred) {
  int max_nodes = (1 << (max_depth + 1)) - 1;
  auto* tree = new BFSTree();
  tree->K = K;
  tree->D = D;
  tree->max_depth = max_depth;
  tree->total_nodes = max_nodes;
  tree->is_leaf.assign(max_nodes, 1);
  tree->split_hyp_idx.assign(max_nodes, -1);
  tree->split_threshold.assign(max_nodes, 0.0f);
  tree->leaf_values.assign((size_t)max_nodes * K, 0.0f);
  tree->split_gain.assign(max_nodes, 0.0f);
  tree->split_weights.assign((size_t)max_nodes * D, 0.0f);

  // d_sub: number of features sampled per node for WLS
  int d_sub = std::max(2, std::min(d_sub_max,
                                   (int)std::ceil(std::sqrt((float)D_num))));

  // Pre-compute bundle exclusivity matrix (one-time, O(D²) but only if D≤150)
  std::vector<bool> excl = compute_bundle_matrix(X, D, D_num, sub, Ns);

  std::vector<std::vector<int>> node_samp(max_nodes);
  node_samp[0].assign(sub, sub + Ns);

  for (int depth = 0; depth < max_depth; depth++) {
    int first_node = (1 << depth) - 1, n_at_depth = 1 << depth;

    std::vector<int> active_nodes;
    for (int local = 0; local < n_at_depth; local++) {
      int t = first_node + local;
      if ((int)node_samp[t].size() < 20) continue;
      active_nodes.push_back(t);
    }
    if (active_nodes.empty()) break;

    int n_active = (int)active_nodes.size();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int a_idx = 0; a_idx < n_active; a_idx++) {
      int t_node = active_nodes[a_idx];
      const auto& samp = node_samp[t_node];
      int ns = (int)samp.size();
      int tl = 2 * t_node + 1, tr_node = 2 * t_node + 2;

      // Independent per-node RNG
      std::mt19937 rng(seed ^ (unsigned)t_node ^ (unsigned)(depth * 1000003u));

      // ── Feature subsampling: pick d_sub features ──────────────────────────
      std::vector<int> feat_sub(D_num);
      std::iota(feat_sub.begin(), feat_sub.end(), 0);
      if (d_sub < D_num) {
        std::shuffle(feat_sub.begin(), feat_sub.end(), rng);
        feat_sub.resize(d_sub);
      }

      // ── Instance subsampling for WLS ──────────────────────────────────────
      std::vector<int> wls_samp;
      if (ns > n_wls_max) {
        wls_samp = samp;
        std::shuffle(wls_samp.begin(), wls_samp.end(), rng);
        wls_samp.resize(n_wls_max);
      } else {
        wls_samp = samp;
      }

      // ── Local bundle check for high-D (only among d_sub selected features)
      std::vector<bool> excl_local;
      if (D_num > 150)
        excl_local = local_bundle_check(wls_samp, feat_sub, D, X);

      // ── Block WLS: find best oblique direction for this node ───────────────
      std::vector<float> best_w = node_block_wls(
          wls_samp, feat_sub, D, D_num, X, G, H, K,
          reg_lambda, energy_frac, excl, excl_local);
      if (best_w.empty()) continue;

      // ── Compute Gt/Ht using full node data ────────────────────────────────
      std::vector<float> Gt(K, 0.0f), Ht(K, 0.0f);
      for (int j : samp)
        for (int c = 0; c < K; c++) {
          Gt[c] += G[(size_t)j * K + c];
          Ht[c] += H[(size_t)j * K + c];
        }
      float total_base = 0.0f;
      for (int c = 0; c < K; c++)
        total_base -= 0.5f * Gt[c] * Gt[c] / (Ht[c] + reg_lambda + EPS);

      // ── Project full node data → 1-D histogram scan ───────────────────────
      std::vector<float> node_proj(ns);
      float min_v = 1e30f, max_v = -1e30f;
      for (int si = 0; si < ns; si++) {
        const float* xi = X + (size_t)samp[si] * D;
        float proj = 0.0f;
        for (int f = 0; f < D_num; f++) proj += best_w[f] * xi[f];
        node_proj[si] = proj;
        if (proj < min_v) min_v = proj;
        if (proj > max_v) max_v = proj;
      }
      if (max_v - min_v < 1e-5f) continue;

      std::vector<float> bin_G((size_t)HIST_BINS * K, 0.0f);
      std::vector<float> bin_H((size_t)HIST_BINS * K, 0.0f);
      std::vector<int>   bin_cnt(HIST_BINS, 0);
      float scale = (float)HIST_BINS / (max_v - min_v + EPS);
      for (int si = 0; si < ns; si++) {
        int j = samp[si];
        int b = (int)((node_proj[si] - min_v) * scale);
        if (b < 0) b = 0;
        if (b >= HIST_BINS) b = HIST_BINS - 1;
        bin_cnt[b]++;
        size_t bo = (size_t)b * K;
        for (int c = 0; c < K; c++) {
          bin_G[bo + c] += G[(size_t)j * K + c];
          bin_H[bo + c] += H[(size_t)j * K + c];
        }
      }

      std::vector<float> Gc(K, 0.0f), Hc(K, 0.0f);
      int n_left = 0;
      float best_gain = 0.0f, best_thr = 0.0f;
      bool split_found = false;
      for (int b = 0; b < HIST_BINS - 1; b++) {
        n_left += bin_cnt[b];
        size_t bo = (size_t)b * K;
        for (int c = 0; c < K; c++) {
          Gc[c] += bin_G[bo + c];
          Hc[c] += bin_H[bo + c];
        }
        int n_right = ns - n_left;
        if (n_left < 10 || n_right < 10) continue;

        float Hc_sum = 0.0f, Hr_sum = 0.0f;
        for (int c = 0; c < K; c++) {
          Hc_sum += Hc[c];
          Hr_sum += Ht[c] - Hc[c];
        }
        if (Hc_sum < MIN_CHILD_W || Hr_sum < MIN_CHILD_W) continue;

        float gain = total_base;
        for (int c = 0; c < K; c++) {
          float Gr = Gt[c] - Gc[c], Hr = Ht[c] - Hc[c];
          gain += 0.5f * (Gc[c] * Gc[c] / (Hc[c] + reg_lambda + EPS) +
                          Gr * Gr / (Hr + reg_lambda + EPS));
        }
        if (gain > best_gain) {
          best_gain    = gain;
          best_thr     = min_v + ((float)(b + 1) / HIST_BINS) * (max_v - min_v);
          split_found  = true;
        }
      }

      if (split_found) {
        tree->is_leaf[t_node]         = 0;
        tree->split_threshold[t_node] = best_thr;
        tree->split_gain[t_node]      = best_gain;
        std::copy(best_w.begin(), best_w.end(),
                  tree->split_weights.data() + (size_t)t_node * D);

        std::vector<int> left_sub, right_sub;
        for (int si = 0; si < ns; si++)
          (node_proj[si] < best_thr ? left_sub : right_sub).push_back(samp[si]);
        node_samp[tl]      = std::move(left_sub);
        node_samp[tr_node] = std::move(right_sub);
      }
    }
  }

  for (int t = 0; t < max_nodes; t++) {
    if (!tree->is_leaf[t] || node_samp[t].empty()) continue;
    for (int c = 0; c < K; c++) {
      float Gs = 0.0f, Hs = 0.0f;
      for (int j : node_samp[t]) {
        Gs += G[(size_t)j * K + c];
        Hs += H[(size_t)j * K + c];
      }
      tree->leaf_values[(size_t)t * K + c] = -Gs / (Hs + reg_lambda + EPS);
    }
  }

  if (out_pred) {
    std::memset(out_pred, 0, (size_t)N * K * sizeof(float));
    _salot_route(tree, X, N, out_pred);
  }
  return static_cast<void*>(tree);
}

void salot_predict(void* tree_handle, const float* X, int N, int K,
                   float* out_pred) {
  const BFSTree* tree = static_cast<const BFSTree*>(tree_handle);
  if (!tree) {
    std::memset(out_pred, 0, (size_t)N * K * sizeof(float));
    return;
  }
  std::memset(out_pred, 0, (size_t)N * K * sizeof(float));
  _salot_route(tree, X, N, out_pred);
}

void salot_tree_free(void* tree_handle) {
  delete static_cast<BFSTree*>(tree_handle);
}

}  // extern "C"
