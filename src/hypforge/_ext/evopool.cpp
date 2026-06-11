// evopool.cpp — EvoPool v2: distilled feature-evolution boosting engine
//
// THE LOOP: gradients evolve features; trees are disposable histogram
// routers over the feature population; the tree's node tournaments ARE the
// fitness evaluation; survival = being used.
//
// What was removed relative to v1 (distillation):
//   * UCB bandit (Welford mu/M2, exploration bonus, grace constants) — the
//     tree already runs a full P-way tournament at every node; a separate
//     fitness scan double-counts. Selection is one integer: rounds_unused.
//   * Separate root fitness scan — O(P·ne) per round gone.
//   * Random alpha-blend crossover — replaced by JOINT REFIT on the union
//     support (CD finds the optimal combination; a random convex blend is
//     strictly worse). One less RNG.
//   * Per-node estimation subsampling — superseded by exact histogram
//     subtraction (below).
//
// What was added (power):
//   * Unified evolution operator: every birth is "CD solve on a support
//     set". Birth = SIS support; Mutation = an incumbent's own support,
//     refit on current residuals; Crossover = union of two incumbents'
//     supports, joint refit. One mechanism, three operators.
//   * Newton-residual births: features are evolved from g' = g + h·f_tree
//     (what remains after a FULL step of this round's tree) — the generator
//     targets exactly the structure the current population failed to
//     explain, root and depth-1 niches alike.
//   * Histogram subtraction (LightGBM classic, exact here because honest
//     splitting is gone): only the smaller child is histogrammed; the larger
//     child = parent − smaller, in place. Leaf/internal G,H totals come from
//     histogram column sums — no extra passes over samples.
//
// Determinism: one RNG derived from (seed, round); admission and culling are
// serial and order-stable; OMP loops touch disjoint state.
//
// Memory: (D + max_extra) · N bytes of uint8 codes; one histogram buffer
// per live node on the current level (P·256·(2K+1) floats each).
//
// API (trees are BFSTree* in salot format → salot_predict/salot_tree_free):
//   salot_evo_create(D, D_num, max_extra)
//   salot_evo_round(pool, X, N, G, H, K, sub, Ns, max_depth, reg_lambda,
//                   seed, out_pred) -> tree
//   salot_evo_pool_size / salot_evo_stats / salot_evo_free

#include "bfstree_types.h"
#include "salot_core.h"

namespace {

constexpr float DEDUP_COS = 0.95f;
constexpr int MUT_K = 2;   // incumbents refit (mutation) per round
constexpr int CX_K = 1;    // union-support joint refits (crossover) per round

struct EvoFeature {
  std::vector<std::pair<int, float>> nz;  // sparse direction, ascending index
  std::vector<uint8_t> code;              // AX_BINS codes for all N rows
  float fmin = 0.0f, frange = 0.0f;
  int use_count = 0;
  int rounds_unused = 0;
  bool is_base = false;
};

struct EvoPool {
  int D, D_num, cap_extra;
  int N = -1;
  int round = 0;
  std::vector<EvoFeature> feats;
  EvoPool(int D_, int Dn_, int cap_) : D(D_), D_num(Dn_), cap_extra(cap_) {}
};

void evo_bin_feature(EvoFeature& f, const float* X, int N, int D,
                     const int* sub, int Ns) {
  std::vector<float> proj(N);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++)
    proj[i] = sparse_dot(f.nz, X + (size_t)i * D);

  float mn = 1e30f, mx = -1e30f;
  for (int si = 0; si < Ns; si++) {
    float v = proj[sub[si]];
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  f.fmin = mn;
  f.frange = (mx - mn > 1e-12f) ? (mx - mn) : 0.0f;
  f.code.assign(N, 0);
  if (f.frange == 0.0f) return;
  float scale = (float)AX_BINS / (f.frange + EPS);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++) {
    int b = (int)((proj[i] - mn) * scale);
    if (b < 0) b = 0;
    if (b >= AX_BINS) b = AX_BINS - 1;
    f.code[i] = (uint8_t)b;
  }
}

// Best split gain of a direction on a sample (HIST_BINS local bins).
float candidate_gain(const std::vector<std::pair<int, float>>& nz,
                     const std::vector<int>& samp_e, const float* X, int D,
                     const float* G, const float* H, int K,
                     float reg_lambda) {
  int ne = (int)samp_e.size();
  std::vector<float> Gt(K, 0.0f), Ht(K, 0.0f);
  for (int j : samp_e)
    for (int c = 0; c < K; c++) {
      Gt[c] += G[(size_t)j * K + c];
      Ht[c] += H[(size_t)j * K + c];
    }
  float base = 0.0f;
  for (int c = 0; c < K; c++)
    base -= 0.5f * Gt[c] * Gt[c] / (Ht[c] + reg_lambda + EPS);

  std::vector<float> proj(ne);
  float mn = 1e30f, mx = -1e30f;
  for (int si = 0; si < ne; si++) {
    float v = sparse_dot(nz, X + (size_t)samp_e[si] * D);
    proj[si] = v;
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }
  if (mx - mn < 1e-12f) return 0.0f;

  std::vector<float> bG((size_t)HIST_BINS * K, 0.0f),
      bH((size_t)HIST_BINS * K, 0.0f);
  std::vector<int> bc(HIST_BINS, 0);
  float scale = (float)HIST_BINS / (mx - mn + EPS);
  for (int si = 0; si < ne; si++) {
    int j = samp_e[si];
    int b = (int)((proj[si] - mn) * scale);
    if (b < 0) b = 0;
    if (b >= HIST_BINS) b = HIST_BINS - 1;
    bc[b]++;
    size_t bo = (size_t)b * K;
    for (int c = 0; c < K; c++) {
      bG[bo + c] += G[(size_t)j * K + c];
      bH[bo + c] += H[(size_t)j * K + c];
    }
  }
  std::vector<float> Gc(K, 0.0f), Hc(K, 0.0f);
  int n_left = 0;
  float best = 0.0f;
  for (int b = 0; b < HIST_BINS - 1; b++) {
    n_left += bc[b];
    size_t bo = (size_t)b * K;
    for (int c = 0; c < K; c++) {
      Gc[c] += bG[bo + c];
      Hc[c] += bH[bo + c];
    }
    int n_right = ne - n_left;
    if (n_left < 10 || n_right < 10) continue;
    float Hcs = 0.0f, Hrs = 0.0f;
    for (int c = 0; c < K; c++) {
      Hcs += Hc[c];
      Hrs += Ht[c] - Hc[c];
    }
    if (Hcs < MIN_CHILD_W || Hrs < MIN_CHILD_W) continue;
    float gain = base;
    for (int c = 0; c < K; c++) {
      float Gr = Gt[c] - Gc[c], Hr = Ht[c] - Hc[c];
      gain += 0.5f * (Gc[c] * Gc[c] / (Hc[c] + reg_lambda + EPS) +
                      Gr * Gr / (Hr + reg_lambda + EPS));
    }
    if (gain > best) best = gain;
  }
  float norm = std::abs(base) + EPS;
  return best / norm;  // scale-free relative gain
}

bool is_redundant(const std::vector<std::pair<int, float>>& nz,
                  const std::vector<EvoFeature>& feats) {
  for (const auto& f : feats)
    if (std::abs(sparse_sparse_dot(nz, f.nz)) > DEDUP_COS) return true;
  return false;
}

// Deterministic stride subsample of a region.
std::vector<int> stride_cap(const std::vector<int>& region, int cap) {
  if ((int)region.size() <= cap) return region;
  std::vector<int> out;
  out.reserve(cap);
  double step = (double)region.size() / cap;
  for (int i = 0; i < cap; i++) out.push_back(region[(size_t)(i * step)]);
  return out;
}

// ── Unified evolution operator: CD solve on a support set ───────────────────
// support empty → derive via SIS on the region (birth);
// support = incumbent's dims (mutation) or union of two (crossover).
void evolve_on_support(EvoPool* pool, const std::vector<int>& region,
                       std::vector<int> support, const float* X, int N,
                       const float* G, const float* H, int K,
                       float reg_lambda, const int* sub, int Ns) {
  int D = pool->D, D_num = pool->D_num;
  if ((int)region.size() < 4 * D_SUB_MAX) return;
  std::vector<int> region_e = stride_cap(region, EST_NE_MAX);
  std::vector<int> wls_samp = stride_cap(region, WLS_CAP);
  int best_c = dominant_class(wls_samp, G, K);

  if (support.empty()) {  // birth: SIS-screened support
    std::vector<float> fscore;
    sis_scores(wls_samp, SCREEN_N, X, D, D_num, G, H, K, best_c, reg_lambda,
               fscore);
    int b_sel = std::min(D_SUB_MAX, D_num);
    support.resize(D_num);
    std::iota(support.begin(), support.end(), 0);
    if (b_sel < D_num) {
      std::partial_sort(support.begin(), support.begin() + b_sel,
                        support.end(),
                        [&](int a, int c) { return fscore[a] > fscore[c]; });
      support.resize(b_sel);
    }
  }
  if ((int)support.size() < 2) return;
  std::sort(support.begin(), support.end());
  support.erase(std::unique(support.begin(), support.end()), support.end());
  if ((int)support.size() > D_SUB_MAX) support.resize(D_SUB_MAX);

  std::vector<std::vector<float>> dirs;
  node_cd_candidates(wls_samp, support, D, X, G, H, K, best_c, reg_lambda,
                     dirs);
  for (const auto& w_c : dirs) {
    std::vector<std::pair<int, float>> nz;
    collect_nonzero(w_c.data(), D_num, nz);
    if (nz.size() < 2) continue;
    if (is_redundant(nz, pool->feats)) continue;
    if (candidate_gain(nz, region_e, X, D, G, H, K, reg_lambda) <= 1e-5f)
      continue;
    EvoFeature f;
    f.nz = std::move(nz);
    evo_bin_feature(f, X, N, D, sub, Ns);
    if (f.frange == 0.0f) continue;
    pool->feats.push_back(std::move(f));
  }
}

}  // namespace

extern "C" {

SALOT_API void* salot_evo_create(int D, int D_num, int max_extra) {
  return static_cast<void*>(new EvoPool(D, D_num, max_extra));
}
SALOT_API void salot_evo_free(void* h) { delete static_cast<EvoPool*>(h); }
SALOT_API int salot_evo_pool_size(void* h) {
  return (int)static_cast<EvoPool*>(h)->feats.size();
}
// Diagnostics: per-feature lifetime use count and nonzero count.
SALOT_API void salot_evo_stats(void* h, float* use_out, int* nnz_out) {
  auto* p = static_cast<EvoPool*>(h);
  for (int i = 0; i < (int)p->feats.size(); i++) {
    use_out[i] = (float)p->feats[i].use_count;
    nnz_out[i] = (int)p->feats[i].nz.size();
  }
}

// One boosting round: tree (with histogram subtraction) → credit/culling →
// Newton-residual evolution for the next round.
SALOT_API void* salot_evo_round(void* handle, const float* X, int N,
                                const float* G, const float* H, int K,
                                const int* sub, int Ns, int max_depth,
                                float reg_lambda, unsigned int seed,
                                float* out_pred) {
  auto* pool = static_cast<EvoPool*>(handle);
  int D = pool->D, D_num = pool->D_num;
  pool->round++;

  // ── Founding population: the raw features themselves ──────────────────────
  if (pool->N < 0) {
    pool->N = N;
    pool->feats.resize(D);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4)
#endif
    for (int d = 0; d < D; d++) {
      EvoFeature& f = pool->feats[d];
      f.is_base = true;
      f.nz = {{d, 1.0f}};
      evo_bin_feature(f, X, N, D, sub, Ns);
    }
    // Seed the pool with first-round births from the raw gradients so the
    // very first tree already competes oblique features.
    evolve_on_support(pool, std::vector<int>(sub, sub + Ns), {}, X, N, G, H,
                      K, reg_lambda, sub, Ns);
  }

  int P = (int)pool->feats.size();
  std::vector<uint8_t> codes_flat((size_t)N * P);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++) {
    for (int p = 0; p < P; p++) {
      codes_flat[(size_t)i * P + p] = pool->feats[p].code[i];
    }
  }
  const int STRIDE = 2 * K + 1;
  const size_t HSZ = (size_t)P * AX_BINS * STRIDE;

  // ── Tree over the pool with histogram subtraction ──────────────────────────
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

  std::vector<int> node_pidx(max_nodes, -1);
  std::vector<int> node_bcode(max_nodes, 0);
  std::vector<std::vector<int>> node_samp(max_nodes);
  std::vector<std::vector<float>> node_hist(max_nodes);
  std::vector<float> node_G((size_t)max_nodes * K, 0.0f);
  std::vector<float> node_H((size_t)max_nodes * K, 0.0f);
  std::vector<char> node_live(max_nodes, 0);

  node_samp[0].assign(sub, sub + Ns);
  node_live[0] = 1;
  // Root histogram: one pass over all rows in sub.
  {
    node_hist[0].assign(HSZ, 0.0f);
    float* SALOT_RESTRICT hb = node_hist[0].data();
    for (int si = 0; si < Ns; si++) {
      int j = sub[si];
      const float* SALOT_RESTRICT gj = G + (size_t)j * K;
      const float* SALOT_RESTRICT hj = H + (size_t)j * K;
      const uint8_t* SALOT_RESTRICT cj = codes_flat.data() + (size_t)j * P;
      for (int p = 0; p < P; p++) {
        float* SALOT_RESTRICT slot =
            hb + ((size_t)p * AX_BINS + cj[p]) * STRIDE;
        for (int c = 0; c < K; c++) {
          slot[c] += gj[c];
          slot[K + c] += hj[c];
        }
        slot[2 * K] += 1.0f;
      }
    }
  }

  for (int depth = 0; depth < max_depth; depth++) {
    int first_node = (1 << depth) - 1, n_at_depth = 1 << depth;
    std::vector<int> active_nodes;
    for (int local = 0; local < n_at_depth; local++) {
      int t = first_node + local;
      if (node_live[t] && (int)node_samp[t].size() >= 20)
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
      const float* SALOT_RESTRICT hb = node_hist[t_node].data();

      // Node totals from histogram column 0 (constant-feature columns still
      // accumulate every sample into bin 0, so column sums are exact).
      std::vector<float> Gt(K, 0.0f), Ht(K, 0.0f);
      for (int b = 0; b < AX_BINS; b++) {
        const float* SALOT_RESTRICT slot = hb + (size_t)b * STRIDE;
        for (int c = 0; c < K; c++) {
          Gt[c] += slot[c];
          Ht[c] += slot[K + c];
        }
      }
      for (int c = 0; c < K; c++) {
        node_G[(size_t)t_node * K + c] = Gt[c];
        node_H[(size_t)t_node * K + c] = Ht[c];
      }
      float total_base = 0.0f;
      for (int c = 0; c < K; c++)
        total_base -= 0.5f * Gt[c] * Gt[c] / (Ht[c] + reg_lambda + EPS);

      // ── Tournament: scan every pool feature's histogram ───────────────────
      float best_gain = 0.0f;
      int best_p = -1, best_b = 0;
      std::vector<float> Gc(K), Hc(K);
      for (int p = 0; p < P; p++) {
        if (pool->feats[p].frange == 0.0f) continue;
        const float* SALOT_RESTRICT fbuf = hb + (size_t)p * AX_BINS * STRIDE;
        std::fill(Gc.begin(), Gc.end(), 0.0f);
        std::fill(Hc.begin(), Hc.end(), 0.0f);
        int n_left = 0;
        for (int b = 0; b < AX_BINS - 1; b++) {
          const float* SALOT_RESTRICT slot = fbuf + (size_t)b * STRIDE;
          n_left += (int)slot[2 * K];
          for (int c = 0; c < K; c++) {
            Gc[c] += slot[c];
            Hc[c] += slot[K + c];
          }
          int n_right = ns - n_left;
          if (n_left < 10 || n_right < 10) continue;
          float Hcs = 0.0f, Hrs = 0.0f;
          for (int c = 0; c < K; c++) {
            Hcs += Hc[c];
            Hrs += Ht[c] - Hc[c];
          }
          if (Hcs < MIN_CHILD_W || Hrs < MIN_CHILD_W) continue;
          float gain = total_base;
          for (int c = 0; c < K; c++) {
            float Gr = Gt[c] - Gc[c], Hr = Ht[c] - Hc[c];
            gain += 0.5f * (Gc[c] * Gc[c] / (Hc[c] + reg_lambda + EPS) +
                            Gr * Gr / (Hr + reg_lambda + EPS));
          }
          if (gain > best_gain) {
            best_gain = gain;
            best_p = p;
            best_b = b;
          }
        }
      }
      if (best_p < 0) {
        node_hist[t_node].clear();
        node_hist[t_node].shrink_to_fit();
        continue;
      }

      const EvoFeature& bf = pool->feats[best_p];
      tree->is_leaf[t_node] = 0;
      tree->split_gain[t_node] = best_gain;
      tree->split_hyp_idx[t_node] = best_p;
      tree->split_threshold[t_node] =
          bf.fmin + ((float)(best_b + 1) / AX_BINS) * bf.frange;
      for (const auto& pr : bf.nz)
        tree->split_weights[(size_t)t_node * D + pr.first] = pr.second;
      node_pidx[t_node] = best_p;
      node_bcode[t_node] = best_b;

      // Partition samples by code.
      const uint8_t* SALOT_RESTRICT cp = pool->feats[best_p].code.data();
      std::vector<int> left_sub, right_sub;
      for (int j : samp)
        (cp[j] <= best_b ? left_sub : right_sub).push_back(j);

      // ── Histogram subtraction: build the SMALLER child, derive the larger
      //    in place from the parent buffer. Exact (no honest split, no
      //    per-node subsampling).
      bool left_small = left_sub.size() <= right_sub.size();
      const std::vector<int>& small_samp = left_small ? left_sub : right_sub;
      int t_small = left_small ? tl : tr_node;
      int t_large = left_small ? tr_node : tl;

      if (depth + 1 < max_depth) {
        node_hist[t_small].assign(HSZ, 0.0f);
        float* SALOT_RESTRICT hs = node_hist[t_small].data();
        for (int j : small_samp) {
          const float* SALOT_RESTRICT gj = G + (size_t)j * K;
          const float* SALOT_RESTRICT hj = H + (size_t)j * K;
          const uint8_t* SALOT_RESTRICT cj = codes_flat.data() + (size_t)j * P;
          for (int p = 0; p < P; p++) {
            float* SALOT_RESTRICT slot =
                hs + ((size_t)p * AX_BINS + cj[p]) * STRIDE;
            for (int c = 0; c < K; c++) {
              slot[c] += gj[c];
              slot[K + c] += hj[c];
            }
            slot[2 * K] += 1.0f;
          }
        }
        // larger = parent − smaller, reusing the parent buffer (no alloc)
        float* SALOT_RESTRICT hp = node_hist[t_node].data();
        for (size_t i = 0; i < HSZ; i++) hp[i] -= hs[i];
        node_hist[t_large] = std::move(node_hist[t_node]);
        node_live[t_small] = node_live[t_large] = 1;
      } else {
        node_hist[t_node].clear();
        node_hist[t_node].shrink_to_fit();
      }

      node_samp[tl] = std::move(left_sub);
      node_samp[tr_node] = std::move(right_sub);
    }
  }
  // Free any remaining level histograms; compute leaf-node totals (leaves at
  // the last level never ran the scan, so their totals come from one pass).
  for (int t = 0; t < max_nodes; t++) {
    node_hist[t].clear();
    node_hist[t].shrink_to_fit();
    if (!node_samp[t].empty() && node_G[(size_t)t * K] == 0.0f &&
        node_H[(size_t)t * K] == 0.0f) {
      for (int j : node_samp[t])
        for (int c = 0; c < K; c++) {
          node_G[(size_t)t * K + c] += G[(size_t)j * K + c];
          node_H[(size_t)t * K + c] += H[(size_t)j * K + c];
        }
    }
  }

  // ── Leaves: path smoothing scaled by lambda (off at lambda = 0) ───────────
  {
    std::vector<float> sm((size_t)max_nodes * K, 0.0f);
    std::vector<char> hasv(max_nodes, 0);
    for (int t = 0; t < max_nodes; t++) {
      if (node_samp[t].empty() && t != 0) continue;
      int par = (t - 1) / 2;
      bool use_parent = (t > 0) && hasv[par];
      for (int c = 0; c < K; c++) {
        float Gs = node_G[(size_t)t * K + c], Hs = node_H[(size_t)t * K + c];
        float raw = -Gs / (Hs + reg_lambda + EPS);
        float v = use_parent
                      ? (Hs * raw + reg_lambda * sm[(size_t)par * K + c]) /
                            (Hs + reg_lambda + EPS)
                      : raw;
        sm[(size_t)t * K + c] = v;
        if (tree->is_leaf[t]) tree->leaf_values[(size_t)t * K + c] = v;
      }
      hasv[t] = 1;
    }
  }

  // ── In-sample predictions via codes ────────────────────────────────────────
  if (out_pred) {
    std::memset(out_pred, 0, (size_t)N * K * sizeof(float));
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; i++) {
      int t = 0;
      for (int dep = 0; dep < max_depth; dep++) {
        if (tree->is_leaf[t]) break;
        t = (pool->feats[node_pidx[t]].code[i] <= node_bcode[t]) ? (2 * t + 1)
                                                                 : (2 * t + 2);
      }
      const float* lv = tree->leaf_values.data() + (size_t)t * K;
      float* oi = out_pred + (size_t)i * K;
      for (int k = 0; k < K; k++) oi[k] = lv[k];
    }
  }

  // ── Credit + culling: survival = being used ────────────────────────────────
  {
    std::vector<char> used(P, 0);
    for (int t = 0; t < max_nodes; t++)
      if (node_pidx[t] >= 0) used[node_pidx[t]] = 1;
    for (int p = 0; p < P; p++) {
      if (used[p]) {
        pool->feats[p].use_count++;
        pool->feats[p].rounds_unused = 0;
      } else {
        pool->feats[p].rounds_unused++;
      }
    }
    int cap = D + pool->cap_extra;
    while ((int)pool->feats.size() > cap) {
      int worst = -1, worst_unused = 0, worst_uc = 1 << 30;
      for (int p = 0; p < (int)pool->feats.size(); p++) {
        const EvoFeature& f = pool->feats[p];
        if (f.is_base || f.rounds_unused < 2) continue;
        if (f.rounds_unused > worst_unused ||
            (f.rounds_unused == worst_unused && f.use_count < worst_uc)) {
          worst_unused = f.rounds_unused;
          worst_uc = f.use_count;
          worst = p;
        }
      }
      if (worst < 0) break;
      pool->feats.erase(pool->feats.begin() + worst);
    }
  }

  // ── Evolution for NEXT round: births from the CURRENT gradients ───────────
  //    (A full-step Newton residual g + h·f overshoots relative to the
  //    learner's actual eta-step and breeds features misaligned with the
  //    next round's true residuals — measured regression. The niche regions
  //    below already give "what the tree missed" localization.)
  {
    const float* Gn = G;
    std::vector<int> root_region(sub, sub + Ns);

    // Births: root + depth-1 niches (SIS supports on the residual).
    evolve_on_support(pool, root_region, {}, X, N, Gn, H, K, reg_lambda, sub,
                      Ns);
    if (!tree->is_leaf[0]) {
      // depth-1 and depth-2 niches: routing regions as evolutionary niches
      for (int t : {1, 2, 3, 4, 5, 6})
        if (!node_samp[t].empty())
          evolve_on_support(pool, node_samp[t], {}, X, N, Gn, H, K,
                            reg_lambda, sub, Ns);
    }

    // Mutation + crossover: refit incumbents' supports on the residual.
    std::vector<int> ord;
    for (int p = 0; p < (int)pool->feats.size(); p++)
      if (!pool->feats[p].is_base && pool->feats[p].nz.size() >= 2)
        ord.push_back(p);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) {
      return pool->feats[a].use_count > pool->feats[b].use_count;
    });
    auto support_of = [&](int p) {
      std::vector<int> s;
      for (const auto& pr : pool->feats[p].nz)
        if (pr.first < D_num) s.push_back(pr.first);
      return s;
    };
    for (int m = 0; m < std::min((int)ord.size(), MUT_K); m++)
      evolve_on_support(pool, root_region, support_of(ord[m]), X, N, Gn, H, K,
                        reg_lambda, sub, Ns);
    if ((int)ord.size() >= 2 && CX_K > 0) {
      std::vector<int> uni = support_of(ord[0]);
      std::vector<int> s2 = support_of(ord[1]);
      uni.insert(uni.end(), s2.begin(), s2.end());
      evolve_on_support(pool, root_region, uni, X, N, Gn, H, K, reg_lambda,
                        sub, Ns);
    }
  }

  (void)seed;  // engine is deterministic without per-round RNG state
  return static_cast<void*>(tree);
}

}  // extern "C"
