// oqboost.cpp — OQBoost v9: context-cached, subtraction-based oblique booster
// with native missing-value and categorical handling.
//
// One route, two hyperparameters (max_depth, reg_lambda), fully
// DETERMINISTIC (no RNG anywhere: subsampling is deterministic striding).
//
// v9 over v8 — native NaN + categorical via ONE transformed space x̃:
//   * Numeric NaN → dynamic mean imputation: μ_f computed once per context
//     from the non-missing values; x̃ = μ_f where x is NaN. The imputed
//     matrix Ximp feeds binning, oblique CD/WLS, and routing identically —
//     no special-cased default direction (an oblique w·x has no "side" to
//     shove NaNs to; a consistent value does not poison the WLS panel).
//   * Categorical → gradient-rank target encoding, re-fit EVERY round:
//     categories are sorted by the signed dominant-class Newton score
//     ΣG/(ΣH+λ) (Fisher-optimal ordering for K=2; one-vs-rest on the
//     dominant class otherwise) and replaced by their monotone rank.
//     Ranks enter the axis scan as ordinary binned values AND the oblique
//     w·x̃ as ordinary coordinates. NaN is its own category and is ranked
//     with the rest; unseen test categories fall back to the NaN rank.
//     Per-tree rank maps are stored for prediction-time re-encoding.
//   * dir_cache stays numeric-only: cat ranks reshuffle each round, so a
//     cached direction touching a cat dim would silently change meaning.
//
// Carried over from v8:
//   * Binning context computed ONCE in gf_ctx_create, reused per round
//     (numeric codes static; only cat columns are re-coded per round).
//   * Histogram subtraction (smaller child fresh, larger = parent − smaller).
//   * Lazy A* best-first growth with admissible node potential.
//   * Oblique directions only where ns ≥ OBLIQUE_MIN.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <queue>
#include <random>
#include <unordered_map>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__APPLE__)
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#define GF_HAVE_BLAS 1
#else
#define GF_HAVE_BLAS 0
#endif

#include "oqboost_types.h"
#include "oqboost_core.h"

static constexpr int OBLIQUE_MIN = 64;  // min node size to fit directions

// ─── Binning context (per dataset, reused across all boosting rounds) ───────
struct OQBoostCtx {
  int N = 0, D = 0, D_num = 0, D_cat = 0;
  std::vector<uint8_t> code;            // N·D uint8 bin codes
  std::vector<float> ax_min, ax_range;  // per-feature bin frame
  std::vector<float> Ximp;              // N·D transformed x̃ (numeric static,
                                        // cat columns rewritten per round)
  std::vector<float> col_mean;          // [D_num] numeric impute means μ_f

  // Categorical raw-value dictionaries (static across rounds).
  std::vector<std::unordered_map<int, int>> cat_id;  // raw → dense id
  std::vector<int> cat_card;       // per cat col: n_distinct + 1 (NaN slot)
  std::vector<int32_t> cat_dense;  // N·D_cat dense ids (NaN → card-1)

  // Persistent direction cache
  static constexpr int DIR_CACHE_MAX = 32;
  std::vector<std::vector<float>> dir_cache;  // dense D each
  int dir_cache_next = 0;

  void cache_direction(const std::vector<float>& w) {
    for (int d = D_num; d < D; d++)
      if (w[d] != 0.0f) return;  // touches a cat dim — meaning is per-round
    float nw = 0.0f;
    for (int d = 0; d < D; d++) nw += w[d] * w[d];
    if (nw < 1e-12f) return;
    float norm = std::sqrt(nw);
    for (const auto& c : dir_cache) {
      float dot = 0.0f;
      for (int d = 0; d < D; d++) dot += c[d] * w[d];
      if (std::abs(dot) / norm > 0.95f) return;  // redundant
    }
    std::vector<float> wn(w.begin(), w.begin() + D);
    float inv_norm = 1.0f / norm;
    for (int d = 0; d < D; d++) wn[d] *= inv_norm;
    if ((int)dir_cache.size() < DIR_CACHE_MAX) {
      dir_cache.push_back(std::move(wn));
    } else {
      dir_cache[dir_cache_next] = std::move(wn);
      dir_cache_next = (dir_cache_next + 1) % DIR_CACHE_MAX;
    }
  }
};

static inline int get_node_depth(int t) {
  int depth = 0;
  while (t > 0) {
    t = (t - 1) / 2;
    depth++;
  }
  return depth;
}

// Deterministic stride subsample.
static std::vector<int> stride_cap(const std::vector<int>& v, int cap) {
  if ((int)v.size() <= cap) return v;
  std::vector<int> out;
  out.reserve(cap);
  double step = (double)v.size() / cap;
  for (int i = 0; i < cap; i++) out.push_back(v[(size_t)(i * step)]);
  return out;
}

// Route a TRANSFORMED matrix x̃ (no NaN, cats already rank-encoded).
static void _gf_route(const OQTree* tree, const float* X, int N,
                         float* out_pred) {
  int D = tree->D, K = tree->K;
  int T = tree->total_nodes;
  std::vector<SparseVec> node_nz(T);
  for (int t = 0; t < T; t++) {
    if (tree->is_leaf[t]) continue;
    collect_nonzero_stack(tree->split_weights.data() + (size_t)t * D, D,
                          node_nz[t]);
  }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++) {
    const float* GF_RESTRICT xi = X + (size_t)i * D;
    int t = 0;
    for (int dep = 0; dep < tree->max_depth; dep++) {
      if (tree->is_leaf[t]) break;
      float proj = sparse_dot_stack(node_nz[t], xi);
      int next_t = (proj < tree->split_threshold[t]) ? (2 * t + 1) : (2 * t + 2);
      if (next_t >= T) break;
      t = next_t;
    }
    const float* lv = tree->leaf_values.data() + (size_t)t * K;
    float* oi = out_pred + (size_t)i * K;
    for (int k = 0; k < K; k++) oi[k] = lv[k];
  }
}

extern "C" {

// Pre-bin all features once.
GF_API void* gf_ctx_create(const float* X, int N, int D, int D_num,
                                 const int* sub, int Ns) {
  auto* ctx = new OQBoostCtx();
  ctx->N = N;
  ctx->D = D;
  ctx->D_num = D_num;
  ctx->D_cat = D - D_num;
  ctx->ax_min.assign(D, 0.0f);
  ctx->ax_range.assign(D, 0.0f);
  ctx->col_mean.assign(D_num, 0.0f);
  ctx->Ximp.assign((size_t)N * D, 0.0f);
  ctx->code.assign((size_t)N * D, 0);

  // ── numeric: μ_f, min/max over the non-missing subsample ─────────────────
  std::vector<float> ax_max(D_num, -1e30f);
  std::vector<float> ax_lo(D_num, 1e30f);
  std::vector<double> sum(D_num, 0.0);
  std::vector<int> cnt(D_num, 0);
  for (int si = 0; si < Ns; si++) {
    const float* GF_RESTRICT xi = X + (size_t)sub[si] * D;
    for (int f = 0; f < D_num; f++) {
      float v = xi[f];
      if (std::isnan(v)) continue;
      if (v < ax_lo[f]) ax_lo[f] = v;
      if (v > ax_max[f]) ax_max[f] = v;
      sum[f] += v;
      cnt[f]++;
    }
  }
  std::vector<float> ax_scale(D_num, 0.0f);
  for (int f = 0; f < D_num; f++) {
    ctx->col_mean[f] = (float)(sum[f] / ((double)cnt[f] + EPS));
    if (cnt[f] == 0) {
      ax_lo[f] = 0.0f;
      continue;
    }
    ctx->ax_min[f] = ax_lo[f];
    float range = ax_max[f] - ax_lo[f];
    if (range > 1e-12f) {
      ctx->ax_range[f] = range;
      ax_scale[f] = (float)AX_BINS / (range + EPS);
    }
  }

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++) {
    const float* GF_RESTRICT xi = X + (size_t)i * D;
    float* GF_RESTRICT ti = ctx->Ximp.data() + (size_t)i * D;
    uint8_t* GF_RESTRICT ci = ctx->code.data() + (size_t)i * D;
    for (int f = 0; f < D_num; f++) {
      float v = xi[f];
      if (std::isnan(v)) v = ctx->col_mean[f];
      ti[f] = v;
      if (ctx->ax_range[f] == 0.0f) continue;
      int b = (int)((v - ctx->ax_min[f]) * ax_scale[f]);
      if (b < 0) b = 0;
      if (b >= AX_BINS) b = AX_BINS - 1;
      ci[f] = (uint8_t)b;
    }
  }

  // ── categorical: value dictionary ────────────────────────────────────────
  if (ctx->D_cat > 0) {
    ctx->cat_id.resize(ctx->D_cat);
    ctx->cat_card.assign(ctx->D_cat, 0);
    ctx->cat_dense.assign((size_t)N * ctx->D_cat, 0);
    for (int fc = 0; fc < ctx->D_cat; fc++) {
      int f = D_num + fc;
      std::vector<int> vals;
      vals.reserve(64);
      for (int i = 0; i < N; i++) {
        float v = X[(size_t)i * D + f];
        if (std::isnan(v)) continue;
        vals.push_back((int)std::lrintf(v));
      }
      std::sort(vals.begin(), vals.end());
      vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
      auto& m = ctx->cat_id[fc];
      m.reserve(vals.size() * 2);
      for (int r = 0; r < (int)vals.size(); r++) m[vals[r]] = r;
      int nan_id = (int)vals.size();  // NaN is its own category
      ctx->cat_card[fc] = nan_id + 1;
      for (int i = 0; i < N; i++) {
        float v = X[(size_t)i * D + f];
        if (std::isnan(v)) {
          ctx->cat_dense[(size_t)i * ctx->D_cat + fc] = nan_id;
        } else {
          auto it = m.find((int)std::lrintf(v));
          ctx->cat_dense[(size_t)i * ctx->D_cat + fc] =
              (it != m.end()) ? it->second : nan_id;
        }
      }
    }
  }
  return static_cast<void*>(ctx);
}

GF_API void gf_ctx_free(void* h) { delete static_cast<OQBoostCtx*>(h); }

// ─── gf_build — one boosting round on a binning context ────────────────
GF_API void* gf_build(void* ctx_handle, const float* X, const float* G,
                               const float* H, int K, const int* sub, int Ns,
                               int max_depth, float reg_lambda,
                               float inherited_rp_ratio, float mutation_rate,
                               float mutation_strength, int seed, float* out_pred) {

  (void)X;
  auto* ctx = static_cast<OQBoostCtx*>(ctx_handle);
  const int D = ctx->D, D_num = ctx->D_num, D_cat = ctx->D_cat, N = ctx->N;
  const float* GF_RESTRICT Xt = ctx->Ximp.data();
  const int STRIDE = 2 * K + 1;
  const size_t HSZ = (size_t)D * AX_BINS * STRIDE;



  const int internal_depth = std::min(2 * max_depth, 22);
  const int max_leaves = 1 << max_depth;
  int max_nodes = (1 << (internal_depth + 1)) - 1;
  auto* tree = new OQTree();
  tree->K = K;
  tree->D = D;
  tree->D_num = D_num;
  tree->max_depth = internal_depth;
  tree->total_nodes = max_nodes;
  tree->is_leaf.assign(max_nodes, 1);
  tree->split_hyp_idx.assign(max_nodes, -1);
  tree->split_threshold.assign(max_nodes, 0.0f);
  tree->leaf_values.assign((size_t)max_nodes * K, 0.0f);
  tree->split_gain.assign(max_nodes, 0.0f);
  tree->split_weights.assign((size_t)max_nodes * D, 0.0f);
  tree->na_means.assign(D, 0.0f);
  std::copy(ctx->col_mean.begin(), ctx->col_mean.end(), tree->na_means.begin());

  // ── per-round categorical re-encoding ────────────────────────────────────
  if (D_cat > 0) {
    tree->cat_ranks.assign(D_cat, {});
    int kdom = 0;
    {
      float best_mass = -1.0f;
      for (int c = 0; c < K; c++) {
        float mcl = 0.0f;
        for (int si = 0; si < Ns; si++)
          mcl += std::abs(G[(size_t)sub[si] * K + c]);
        if (mcl > best_mass) {
          best_mass = mcl;
          kdom = c;
        }
      }
    }
    for (int fc = 0; fc < D_cat; fc++) {
      int f = D_num + fc;
      int card = ctx->cat_card[fc];
      if (card <= 1) {
        ctx->ax_range[f] = 0.0f;
        continue;
      }
      std::vector<double> Gs(card, 0.0), Hs(card, 0.0);
      for (int si = 0; si < Ns; si++) {
        int i = sub[si];
        int id = ctx->cat_dense[(size_t)i * D_cat + fc];
        Gs[id] += G[(size_t)i * K + kdom];
        Hs[id] += H[(size_t)i * K + kdom];
      }
      std::vector<float> score(card);
      for (int id = 0; id < card; id++)
        score[id] = (float)(Gs[id] / (Hs[id] + reg_lambda + EPS));
      std::vector<int> ord(card);
      std::iota(ord.begin(), ord.end(), 0);
      std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        if (score[a] != score[b]) return score[a] < score[b];
        return a < b;
      });
      std::vector<float> rank_of(card);
      for (int r = 0; r < card; r++) rank_of[ord[r]] = (float)r;

      ctx->ax_min[f] = 0.0f;
      ctx->ax_range[f] = (float)(card - 1);
      float scale = (float)AX_BINS / ((float)(card - 1) + EPS);
      float* GF_RESTRICT Xw = ctx->Ximp.data();
      uint8_t* GF_RESTRICT cw = ctx->code.data();
      const int32_t* GF_RESTRICT cd = ctx->cat_dense.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int i = 0; i < N; i++) {
        float r = rank_of[cd[(size_t)i * D_cat + fc]];
        Xw[(size_t)i * D + f] = r;
        int b = (int)(r * scale);
        if (b >= AX_BINS) b = AX_BINS - 1;
        cw[(size_t)i * D + f] = (uint8_t)b;
      }
      auto& rk = tree->cat_ranks[fc];
      rk.reserve(ctx->cat_id[fc].size() * 2);
      for (const auto& kv : ctx->cat_id[fc]) rk[kv.first] = rank_of[kv.second];
      tree->na_means[f] = rank_of[card - 1];
    }
  }
  const uint8_t* GF_RESTRICT code = ctx->code.data();

  // Histogram accumulation lambda (dynamic hybrid parallel)
  auto accumulate_hist = [&](const int* rows, int nr, float* GF_RESTRICT hb,
                             float* node_P_out) {
    double P_acc = 0.0;
#ifdef _OPENMP
    if (nr >= 2048) {
      int nthreads = omp_get_max_threads();
      if (D >= nthreads && D >= 2) {
        // Block-wise Feature-parallelism
        int block_size = std::max(1, D / nthreads);
#pragma omp parallel for schedule(static)
        for (int fg = 0; fg < D; fg += block_size) {
          int f_end = std::min(fg + block_size, D);
          for (int f = fg; f < f_end; f++) {
            float* GF_RESTRICT slot_f = hb + (size_t)f * AX_BINS * STRIDE;
            std::memset(slot_f, 0, AX_BINS * STRIDE * sizeof(float));
          }
          for (int si = 0; si < nr; si++) {
            int j = rows[si];
            const float* GF_RESTRICT gj = G + (size_t)j * K;
            const float* GF_RESTRICT hj = H + (size_t)j * K;
            for (int f = fg; f < f_end; f++) {
              int b = code[(size_t)j * D + f];
              float* GF_RESTRICT slot = hb + ((size_t)f * AX_BINS + b) * STRIDE;
              for (int c = 0; c < K; c++) {
                slot[c] += gj[c];
                slot[K + c] += hj[c];
              }
              slot[2 * K] += 1.0f;
            }
          }
        }
      } else {
        // Sample-parallelism with thread-local histograms (efficient for small D)
        std::vector<float> local_hists((size_t)nthreads * HSZ, 0.0f);
        std::vector<double> tP(nthreads, 0.0);
#pragma omp parallel num_threads(nthreads)
        {
          int tid = omp_get_thread_num();
          float* GF_RESTRICT lb = local_hists.data() + (size_t)tid * HSZ;
          double lp = 0.0;
#pragma omp for schedule(static) nowait
          for (int si = 0; si < nr; si++) {
            int j = rows[si];
            const uint8_t* GF_RESTRICT cj = code + (size_t)j * D;
            const float* GF_RESTRICT gj = G + (size_t)j * K;
            const float* GF_RESTRICT hj = H + (size_t)j * K;
            for (int f = 0; f < D; f++) {
              float* GF_RESTRICT slot = lb + ((size_t)f * AX_BINS + cj[f]) * STRIDE;
              for (int c = 0; c < K; c++) {
                slot[c] += gj[c];
                slot[K + c] += hj[c];
              }
              slot[2 * K] += 1.0f;
            }
            if (node_P_out) {
              for (int c = 0; c < K; c++) {
                lp += 0.5 * (double)gj[c] * gj[c] / ((double)hj[c] + reg_lambda + EPS);
              }
            }
          }
          tP[tid] = lp;
        }
#pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < (int64_t)HSZ; i++) {
          float s = 0.0f;
          for (int t = 0; t < nthreads; t++) {
            s += local_hists[(size_t)t * HSZ + i];
          }
          hb[i] = s;
        }
        if (node_P_out) {
          for (int t = 0; t < nthreads; t++) P_acc += tP[t];
          *node_P_out = (float)P_acc;
        }
        return;
      }
      if (node_P_out) {
#pragma omp parallel for reduction(+:P_acc) schedule(static)
        for (int si = 0; si < nr; si++) {
          int j = rows[si];
          const float* GF_RESTRICT gj = G + (size_t)j * K;
          const float* GF_RESTRICT hj = H + (size_t)j * K;
          for (int c = 0; c < K; c++) {
            P_acc += 0.5 * (double)gj[c] * gj[c] / ((double)hj[c] + reg_lambda + EPS);
          }
        }
        *node_P_out = (float)P_acc;
      }
      return;
    }
#endif
    // Fallback to single-threaded accumulation
    std::memset(hb, 0, HSZ * sizeof(float));
    for (int si = 0; si < nr; si++) {
      int j = rows[si];
      const uint8_t* GF_RESTRICT cj = code + (size_t)j * D;
      const float* GF_RESTRICT gj = G + (size_t)j * K;
      const float* GF_RESTRICT hj = H + (size_t)j * K;
      for (int f = 0; f < D; f++) {
        float* GF_RESTRICT slot = hb + ((size_t)f * AX_BINS + cj[f]) * STRIDE;
        for (int c = 0; c < K; c++) {
          slot[c] += gj[c];
          slot[K + c] += hj[c];
        }
        slot[2 * K] += 1.0f;
      }
      if (node_P_out) {
        for (int c = 0; c < K; c++) {
          P_acc += 0.5 * (double)gj[c] * gj[c] / ((double)hj[c] + reg_lambda + EPS);
        }
      }
    }
    if (node_P_out) *node_P_out = (float)P_acc;
  };

  std::vector<std::vector<int>> node_samp(max_nodes);
  std::vector<std::vector<float>> node_hist(max_nodes);
  std::vector<std::vector<float>> hist_pool;

  auto get_hist = [&]() -> std::vector<float> {
    if (!hist_pool.empty()) {
      auto h = std::move(hist_pool.back());
      hist_pool.pop_back();
      return h;
    }
    return std::vector<float>(HSZ);
  };

  auto recycle_hist = [&](std::vector<float>& h) {
    if (h.size() == HSZ) {
      hist_pool.push_back(std::move(h));
    }
    h.clear();
  };

  std::vector<float> node_G((size_t)max_nodes * K, 0.0f);
  std::vector<float> node_H((size_t)max_nodes * K, 0.0f);
  std::vector<float> node_P(max_nodes, 0.0f);
  std::vector<char> node_has_tot(max_nodes, 0);

  std::vector<float> cand_gain(max_nodes, 0.0f);
  std::vector<float> cand_thr(max_nodes, 0.0f);
  std::vector<int> cand_axis(max_nodes, -1);
  std::vector<int> cand_bcode(max_nodes, 0);
  std::vector<std::vector<float>> cand_w(max_nodes);
  std::vector<std::vector<float>> cand_proj(max_nodes);

  std::vector<char> oblique_done(max_nodes, 0);

  // ── Phase 1: Axis scan ───────────────────────────────────────────────────
  auto eval_axis = [&](int t) -> float {
    int ns = (int)node_samp[t].size();
    const float* GF_RESTRICT hb = node_hist[t].data();

    std::vector<float> Gt(K, 0.0f), Ht(K, 0.0f);
    for (int b = 0; b < AX_BINS; b++) {
      const float* GF_RESTRICT slot = hb + (size_t)b * STRIDE;
      for (int c = 0; c < K; c++) {
        Gt[c] += slot[c];
        Ht[c] += slot[K + c];
      }
    }
    for (int c = 0; c < K; c++) {
      node_G[(size_t)t * K + c] = Gt[c];
      node_H[(size_t)t * K + c] = Ht[c];
    }
    node_has_tot[t] = 1;
    float total_base = 0.0f;
    for (int c = 0; c < K; c++)
      total_base -= 0.5f * Gt[c] * Gt[c] / (Ht[c] + reg_lambda + EPS);

    float best_gain = 0.0f, best_thr = 0.0f;
    int best_axis = -1, best_axis_b = 0;

    // NOTE: an `if (D * K >= 192)` serialization guard was A/B-tested here
    // (2026-06-12) and measured ~7% SLOWER overall — libomp's spin-wait
    // keeps workers hot, so even these small regions win parallel. The
    // profiler's dominant __psynch_cvwait is parked/spinning workers, not
    // lost wall time. Don't re-add work-size guards without an A/B.
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      float l_gain = 0.0f, l_thr = 0.0f;
      int l_axis = -1, l_b = 0;
      std::vector<float> Gc(K), Hc(K);
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
      for (int f = 0; f < D; f++) {
        if (ctx->ax_range[f] == 0.0f) continue;
        const float* GF_RESTRICT fbuf = hb + (size_t)f * AX_BINS * STRIDE;
        std::fill(Gc.begin(), Gc.end(), 0.0f);
        std::fill(Hc.begin(), Hc.end(), 0.0f);
        int n_left = 0;
        for (int b = 0; b < AX_BINS - 1; b++) {
          const float* GF_RESTRICT slot = fbuf + (size_t)b * STRIDE;
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
          if (gain > l_gain || (gain == l_gain && l_axis >= 0 && f < l_axis)) {
            l_gain = gain;
            l_axis = f;
            l_b = b;
            l_thr =
                ctx->ax_min[f] + ((float)(b + 1) / AX_BINS) * ctx->ax_range[f];
          }
        }
      }
#ifdef _OPENMP
#pragma omp critical
#endif
      {
        if (l_axis >= 0 &&
            (l_gain > best_gain ||
             (l_gain == best_gain && (best_axis < 0 || l_axis < best_axis)))) {
          best_gain = l_gain;
          best_axis = l_axis;
          best_axis_b = l_b;
          best_thr = l_thr;
        }
      }
    }

    cand_gain[t] = best_gain;
    cand_thr[t] = best_thr;
    cand_axis[t] = best_axis;
    cand_bcode[t] = best_axis_b;
    cand_proj[t].clear();
    if (best_axis >= 0) {
      cand_w[t].assign(D, 0.0f);
      cand_w[t][best_axis] = 1.0f;
    } else {
      cand_w[t].clear();
    }
    return best_gain;
  };

  // ── Phase 2: Oblique scan ────────────────────────────────────────────────
  auto eval_oblique = [&](int t) -> float {
    const auto& samp = node_samp[t];
    int ns = (int)samp.size();
    if (ns < OBLIQUE_MIN) return cand_gain[t];
    std::vector<float> Gt(K), Ht(K);
    for (int c = 0; c < K; c++) {
      Gt[c] = node_G[(size_t)t * K + c];
      Ht[c] = node_H[(size_t)t * K + c];
    }
    float total_base = 0.0f;
    for (int c = 0; c < K; c++)
      total_base -= 0.5f * Gt[c] * Gt[c] / (Ht[c] + reg_lambda + EPS);

    std::vector<int> samp_e = stride_cap(samp, EST_NE_MAX);
    int ne = (int)samp_e.size();
    bool exact_eval = (ne == ns);
    std::vector<float> eGt(K, 0.0f), eHt(K, 0.0f);
    if (exact_eval) {
      eGt = Gt;
      eHt = Ht;
    } else {
      for (int j : samp_e) {
        for (int c = 0; c < K; c++) {
          eGt[c] += G[(size_t)j * K + c];
          eHt[c] += H[(size_t)j * K + c];
        }
      }
    }
    float e_base = 0.0f;
    for (int c = 0; c < K; c++)
      e_base -= 0.5f * eGt[c] * eGt[c] / (eHt[c] + reg_lambda + EPS);

    int kdom = dominant_class(samp_e, G, K);

    // Gather the evaluation panel contiguously once: every later pass (SIS
    // accumulation, all candidate projections) then walks Xe sequentially
    // instead of striding through Xt rows scattered across the full matrix.
    std::vector<float> Xe((size_t)ne * D);
    for (int i = 0; i < ne; i++) {
      std::memcpy(Xe.data() + (size_t)i * D, Xt + (size_t)samp_e[i] * D,
                  (size_t)D * sizeof(float));
    }

    std::vector<float> cg_s(D, 0.0f), add_s(D, 0.0f);
    for (int i = 0; i < ne; i++) {
      const float* GF_RESTRICT xi = Xe.data() + (size_t)i * D;
      int idx = samp_e[i];
      float gi = G[(size_t)idx * K + kdom];
      float hi = H[(size_t)idx * K + kdom];
      for (int d = 0; d < D; d++) {
        cg_s[d] += xi[d] * gi;
        add_s[d] += hi * xi[d] * xi[d];
      }
    }
    std::vector<float> fscore(D, 0.0f);
    for (int d = 0; d < D; d++) {
      fscore[d] = std::abs(cg_s[d]) / std::sqrt(add_s[d] + reg_lambda + EPS);
    }

    std::vector<std::vector<float>> dirs;
    std::mt19937 rng(seed + t);

    // OQB_GAIN_LOG (research instrumentation, THEORY.md §1 open task):
    // when set to a file path, append one CSV line per evaluated candidate:
    //   node_seq,depth,ns,family,gain
    // family ∈ {x axis, a/b/c inherited strategies, p pobs block, k cache}.
    // Default off — production behavior untouched.
    static FILE* gain_log = [] {
      const char* p = std::getenv("OQB_GAIN_LOG");
      return p ? std::fopen(p, "a") : (FILE*)nullptr;
    }();
    std::vector<char> fam;

    std::vector<float> prob(D);
    for (int d = 0; d < D; d++) {
      prob[d] = fscore[d] + 1e-6f;
    }
    std::discrete_distribution<int> feat_dist(prob.begin(), prob.end());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // pobs_sis candidates (THEORY.md POBS study, merged 2026-06-12): sample
    // an SIS-weighted support S (|S| ≈ √D), build an exact Haar-orthogonal
    // |S|×|S| block ON S via Gram-Schmidt. Sparse (routing-compatible),
    // exactly orthogonal (zero within-block candidate correlation → every
    // draw contributes full marginal tail mass), gradient-informed. Fills
    // the diversity slots (root pool, n_global) plus a fixed 8-candidate
    // slice of every node's inherited budget — the per-node injection is
    // what passed the deployment protocol (root-only did not).
    auto fill_pobs = [&](int count) {
      if (count <= 0) return;
      int m = (int)std::lround(std::sqrt((float)D));
      m = std::max(2, std::min({m, D_SUB_MAX, D}));
      std::normal_distribution<float> gauss(0.0f, 1.0f);
      int target = (int)dirs.size() + count;
      while ((int)dirs.size() < target) {
        // Fresh SIS-weighted support per block (covers varied subspaces).
        std::vector<int> S;
        std::vector<uint8_t> used(D, 0);
        for (int attempt = 0; attempt < 50 * m && (int)S.size() < m;
             attempt++) {
          int f = feat_dist(rng);
          if (!used[f]) {
            used[f] = 1;
            S.push_back(f);
          }
        }
        int mm = (int)S.size();
        if (mm < 2) break;
        // Gaussian mm×mm + modified Gram-Schmidt → Haar-orthogonal rows.
        std::vector<std::vector<float>> B(mm, std::vector<float>(mm));
        for (auto& row : B)
          for (auto& v : row) v = gauss(rng);
        int emitted = 0;
        for (int i = 0; i < mm && (int)dirs.size() < target; i++) {
          for (int j = 0; j < i; j++) {
            float dot = 0.0f;
            for (int d2 = 0; d2 < mm; d2++) dot += B[i][d2] * B[j][d2];
            for (int d2 = 0; d2 < mm; d2++) B[i][d2] -= dot * B[j][d2];
          }
          float n2 = 0.0f;
          for (int d2 = 0; d2 < mm; d2++) n2 += B[i][d2] * B[i][d2];
          if (n2 < 1e-12f) continue;
          float inv = 1.0f / std::sqrt(n2);
          for (int d2 = 0; d2 < mm; d2++) B[i][d2] *= inv;
          std::vector<float> w(D, 0.0f);
          for (int d2 = 0; d2 < mm; d2++) w[S[d2]] = B[i][d2];
          dirs.push_back(std::move(w));
          fam.push_back('p');
          emitted++;
        }
        if (emitted == 0) break;  // degenerate node, avoid spinning
      }
    };

    bool has_parent = (t > 0);
    int par = has_parent ? (t - 1) / 2 : -1;
    SparseVec parent_nz;
    parent_nz.size = 0;
    if (has_parent) {
      collect_nonzero_stack(tree->split_weights.data() + (size_t)par * D, D,
                            parent_nz);
    }

    // Depth-adaptive pool budget (THEORY.md §2/P4-refined, merged
    // 2026-06-12): a node at depth d estimates directions from n/2^d
    // samples — at depth ≥ 3 the gradient statistics are noise-dominated,
    // and each tree LEVEL costs the same total (per-level sample counts sum
    // to N), so a uniform budget overspends exactly where extra candidates
    // buy noise. Cut deep (8) and reinvest shallow (64): order-statistics
    // says to widen the tournament where estimates are reliable and node
    // impact is largest. Cut-only (no reinvest) regressed covtype — the two
    // halves are a package.
    int depth_t = get_node_depth(t);
    int pool_budget = (depth_t <= 2) ? 64 : 8;

    int n_inherited = (int)std::round((float)pool_budget * inherited_rp_ratio);
    if (n_inherited < 0) n_inherited = 0;
    if (n_inherited > pool_budget) n_inherited = pool_budget;
    int n_global = pool_budget - n_inherited;

    // Carve a fixed pobs slice out of the inherited budget so every node
    // sees orthogonal-diverse candidates, not just the root. OQB_POBS=0
    // disables the carve (the inherited budget reverts to all A/B/C) for
    // A/B comparison; the root pool stays pobs either way (the legacy root
    // generator was removed and the root slot measured neutral).
    static const bool pobs_on = [] {
      const char* e = std::getenv("OQB_POBS");
      return !(e && std::atoi(e) == 0);
    }();
    int n_pobs_extra = pobs_on ? std::min(8, n_inherited) : 0;
    n_inherited -= n_pobs_extra;

    if (has_parent && parent_nz.size > 0) {
      // Inherited mutation strategies (A/B/C) — the original production
      // design. Research-impl ablations questioned A/B, but transplanting
      // that finding regressed the real tuned benchmarks; until the
      // mechanism is understood theoretically the proven configuration
      // stays (see research/FINDINGS.md).
      float local_mutation_rate =
          mutation_rate / std::sqrt(1.0f + (float)depth_t);
      float local_mutation_strength =
          mutation_strength / (1.0f + (float)depth_t);

      for (int r = 0; r < n_inherited; r++) {
        float strategy_draw = dist(rng);
        bool do_strategy_a = false;
        bool do_strategy_b = false;
        bool do_strategy_c = false;

        // a:37.5% b:37.5% c:25%. A data-optimal per-node-E[max] reallocation
        // (a:6.25%) was tried and REJECTED on the deployment protocol —
        // Strategy A's ensemble value is invisible in per-node gains
        // (THEORY.md §1 allocation study). This split has survived three
        // challenges; do not reweight it without ensemble-level evidence.
        if (!ctx->dir_cache.empty()) {
          if (strategy_draw < 0.375f) {
            do_strategy_a = (parent_nz.size > 1);
            if (!do_strategy_a) do_strategy_b = true;
          } else if (strategy_draw < 0.75f) {
            do_strategy_b = true;
          } else {
            do_strategy_c = true;
          }
        } else {
          if (strategy_draw < 0.5f) {
            do_strategy_a = (parent_nz.size > 1);
            if (!do_strategy_a) do_strategy_b = true;
          } else {
            do_strategy_b = true;
          }
        }

        std::vector<float> w_rand(D, 0.0f);

        if (do_strategy_c) {
          std::uniform_int_distribution<int> cache_dist(
              0, (int)ctx->dir_cache.size() - 1);
          const auto& cached_w = ctx->dir_cache[cache_dist(rng)];
          float alpha = dist(rng) * 0.6f + 0.2f;
          float one_minus_alpha = 1.0f - alpha;
          for (int d = 0; d < D; d++) {
            w_rand[d] = one_minus_alpha * cached_w[d];
          }
          for (int i_nz = 0; i_nz < parent_nz.size; i_nz++) {
            int d = parent_nz.indices[i_nz];
            w_rand[d] += alpha * parent_nz.values[i_nz];
          }
        } else {
          for (int i_nz = 0; i_nz < parent_nz.size; i_nz++) {
            w_rand[parent_nz.indices[i_nz]] = parent_nz.values[i_nz];
          }

          if (do_strategy_a) {
            std::uniform_int_distribution<int> parent_idx_dist(
                0, parent_nz.size - 1);
            int idx1 = parent_idx_dist(rng);
            float s1 =
                dist(rng) * 2.0f * local_mutation_rate - local_mutation_rate;
            w_rand[parent_nz.indices[idx1]] *= (1.0f + s1);

            if (parent_nz.size > 2 && dist(rng) < 0.5f) {
              int idx2 = parent_idx_dist(rng);
              while (idx2 == idx1) idx2 = parent_idx_dist(rng);
              float s2 =
                  dist(rng) * 2.0f * local_mutation_rate - local_mutation_rate;
              w_rand[parent_nz.indices[idx2]] *= (1.0f + s2);
            }
          } else if (do_strategy_b) {
            std::vector<int> candidate_feats;
            std::vector<float> candidate_probs;
            for (int d = 0; d < D; d++) {
              bool in_parent = false;
              for (int i_nz = 0; i_nz < parent_nz.size; i_nz++) {
                if (parent_nz.indices[i_nz] == d) {
                  in_parent = true;
                  break;
                }
              }
              if (!in_parent) {
                candidate_feats.push_back(d);
                candidate_probs.push_back(prob[d]);
              }
            }

            if (!candidate_feats.empty()) {
              std::discrete_distribution<int> new_feat_dist(
                  candidate_probs.begin(), candidate_probs.end());
              int idx_feat = new_feat_dist(rng);
              int f_new = candidate_feats[idx_feat];
              float sign = (cg_s[f_new] >= 0.0f) ? -1.0f : 1.0f;
              w_rand[f_new] = sign * local_mutation_strength;
            } else if (parent_nz.size == 1) {
              int f_new = (parent_nz.indices[0] + 1) % D;
              w_rand[f_new] = 0.1f;
            }
          }
        }

        float norm = 0.0f;
        for (int d = 0; d < D; d++) norm += w_rand[d] * w_rand[d];
        norm = std::sqrt(norm);
        if (norm > 1e-12f) {
          for (int d = 0; d < D; d++) w_rand[d] /= norm;
          dirs.push_back(std::move(w_rand));
        } else {
          std::vector<float> w_fallback(D, 0.0f);
          w_fallback[parent_nz.indices[0]] = parent_nz.values[0];
          dirs.push_back(std::move(w_fallback));
        }
        fam.push_back(do_strategy_c ? 'c' : (do_strategy_a ? 'a' : 'b'));
      }

      // Diversity slot of the inherited branch (nonzero when
      // inherited_rp_ratio < 1): pobs blocks.
      fill_pobs(n_global);

    } else {
      // Root/no-parent pool: all pobs blocks.
      fill_pobs(pool_budget);
    }

    if (n_pobs_extra > 0) fill_pobs(n_pobs_extra);

    // Cached directions enter the candidate batch directly: with the batched
    // panel projections below, evaluating all of them exactly costs less than
    // the old approximate 512-sample pre-screen pass did, and removes its
    // sampling noise.
    for (const auto& cw : ctx->dir_cache) {
      dirs.push_back(cw);
      fam.push_back('k');
    }

    float ob_gain = 0.0f, ob_thr = 0.0f;
    int ob_idx = -1;
    const int n_dirs = (int)dirs.size();
    std::vector<float> dir_gain(n_dirs, 0.0f), dir_thr(n_dirs, 0.0f);

    // Routing truncates every direction to its first D_SUB_MAX nonzeros
    // (collect_nonzero_stack); sanitize candidates the same way up front so
    // the batched dense projections match what routing will later compute.
    for (auto& wv : dirs) {
      int nz = 0;
      for (int d = 0; d < D; d++) {
        if (wv[d] != 0.0f) {
          if (nz >= D_SUB_MAX) wv[d] = 0.0f;
          else nz++;
        }
      }
    }

    // Batched candidate projections over the contiguous panel:
    // Pj[di·ne + si] = dirs[di]·Xe[si]. One GEMM on the AMX/BLAS path
    // replaces n_dirs scattered sparse-dot sweeps.
    std::vector<float> Pj((size_t)n_dirs * ne);
#if GF_HAVE_BLAS
    {
      std::vector<float> Wd((size_t)n_dirs * D);
      for (int di = 0; di < n_dirs; di++)
        std::memcpy(Wd.data() + (size_t)di * D, dirs[di].data(),
                    (size_t)D * sizeof(float));
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_dirs, ne, D,
                  1.0f, Wd.data(), D, Xe.data(), D, 0.0f, Pj.data(), ne);
    }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n_dirs > 1)
#endif
    for (int di = 0; di < n_dirs; di++) {
      SparseVec sv;
      collect_nonzero_stack(dirs[di].data(), D, sv);
      float* GF_RESTRICT prow = Pj.data() + (size_t)di * ne;
      for (int si = 0; si < ne; si++)
        prow[si] = sparse_dot_stack(sv, Xe.data() + (size_t)si * D);
    }
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (n_dirs > 1)
#endif
    for (int di = 0; di < n_dirs; di++) {
      SparseVec sv;
      collect_nonzero_stack(dirs[di].data(), D, sv);
      if (sv.size <= 1) continue;

      const float* GF_RESTRICT proj_e = Pj.data() + (size_t)di * ne;
      float min_v = 1e30f, max_v = -1e30f;
      for (int si = 0; si < ne; si++) {
        float proj = proj_e[si];
        if (proj < min_v) min_v = proj;
        if (proj > max_v) max_v = proj;
      }
      if (max_v - min_v < 1e-12f) continue;

      static constexpr int K_MAX = 32;
      float bin_G[64 * K_MAX];
      float bin_H[64 * K_MAX];
      float* bin_G_ptr = bin_G;
      float* bin_H_ptr = bin_H;
      std::vector<float> heap_G, heap_H;
      if (K > K_MAX) {
        heap_G.assign((size_t)HIST_BINS * K, 0.0f);
        heap_H.assign((size_t)HIST_BINS * K, 0.0f);
        bin_G_ptr = heap_G.data();
        bin_H_ptr = heap_H.data();
      } else {
        std::memset(bin_G, 0, HIST_BINS * K * sizeof(float));
        std::memset(bin_H, 0, HIST_BINS * K * sizeof(float));
      }
      int bin_cnt[HIST_BINS] = {0};

      float scale = (float)HIST_BINS / (max_v - min_v + EPS);
      for (int si = 0; si < ne; si++) {
        int j = samp_e[si];
        int b = (int)((proj_e[si] - min_v) * scale);
        if (b < 0) b = 0;
        if (b >= HIST_BINS) b = HIST_BINS - 1;
        bin_cnt[b]++;
        size_t bo = (size_t)b * K;
        for (int c = 0; c < K; c++) {
          bin_G_ptr[bo + c] += G[(size_t)j * K + c];
          bin_H_ptr[bo + c] += H[(size_t)j * K + c];
        }
      }

      std::vector<float> lGc(K, 0.0f), lHc(K, 0.0f);
      int n_left = 0;
      float d_gain = 0.0f, d_thr = 0.0f;
      for (int b = 0; b < HIST_BINS - 1; b++) {
        n_left += bin_cnt[b];
        size_t bo = (size_t)b * K;
        for (int c = 0; c < K; c++) {
          lGc[c] += bin_G_ptr[bo + c];
          lHc[c] += bin_H_ptr[bo + c];
        }
        int n_right = ne - n_left;
        if (n_left < 10 || n_right < 10) continue;
        float Hcs = 0.0f, Hrs = 0.0f;
        for (int c = 0; c < K; c++) {
          Hcs += lHc[c];
          Hrs += eHt[c] - lHc[c];
        }
        if (Hcs < MIN_CHILD_W || Hrs < MIN_CHILD_W) continue;
        float gain = e_base;
        for (int c = 0; c < K; c++) {
          float Gr = eGt[c] - lGc[c], Hr = eHt[c] - lHc[c];
          gain += 0.5f * (lGc[c] * lGc[c] / (lHc[c] + reg_lambda + EPS) +
                          Gr * Gr / (Hr + reg_lambda + EPS));
        }
        if (gain > d_gain) {
          d_gain = gain;
          d_thr = min_v + ((float)(b + 1) / HIST_BINS) * (max_v - min_v);
        }
      }
      if (d_gain > 0.0f) {
        dir_gain[di] = d_gain;
        dir_thr[di] = d_thr;
      }
    }

    if (gain_log) {
      static int node_seq = 0;
      int nid = node_seq++;
      // Axis-scan best gain for this node (family 'x'; computed on the full
      // node, so its scale differs slightly from the samp_e-estimated pool
      // gains — compare within the pool, use 'x' as the per-node baseline).
      std::fprintf(gain_log, "%d,%d,%d,x,%.6g\n", nid, depth_t, ns,
                   cand_gain[t]);
      for (int di = 0; di < n_dirs; di++)
        std::fprintf(gain_log, "%d,%d,%d,%c,%.6g\n", nid, depth_t, ns,
                     di < (int)fam.size() ? fam[di] : '?', dir_gain[di]);
    }

    for (int di = 0; di < n_dirs; di++) {
      if (dir_gain[di] > ob_gain) {
        ob_gain = dir_gain[di];
        ob_thr = dir_thr[di];
        ob_idx = di;
      }
    }
    if (ob_idx < 0) return cand_gain[t];
    SparseVec sv;

    collect_nonzero_stack(dirs[ob_idx].data(), D, sv);
    std::vector<float> proj_full(ns);
    float exact_gain;
    if (exact_eval) {
      // samp_e == samp here, so the batched panel projections are exactly
      // the per-sample projections routing needs.
      std::memcpy(proj_full.data(), Pj.data() + (size_t)ob_idx * ne,
                  (size_t)ne * sizeof(float));
      exact_gain = ob_gain;
    } else {
      std::vector<float> GL(K, 0.0f), HL(K, 0.0f);
      int n_left = 0;
#ifdef _OPENMP
      if (ns >= 8192) {
        int nthreads = omp_get_max_threads();
        std::vector<std::vector<float>> tGL(nthreads, std::vector<float>(K, 0.0f));
        std::vector<std::vector<float>> tHL(nthreads, std::vector<float>(K, 0.0f));
        std::vector<int> tnl(nthreads, 0);
        int actual_threads = nthreads;
#pragma omp parallel num_threads(nthreads)
        {
          int tid = omp_get_thread_num();
          if (tid == 0) actual_threads = omp_get_num_threads();
          float* GF_RESTRICT lGL = tGL[tid].data();
          float* GF_RESTRICT lHL = tHL[tid].data();
          int lnl = 0;
#pragma omp for schedule(static) nowait
          for (int si = 0; si < ns; si++) {
            int j = samp[si];
            float proj = sparse_dot_stack(sv, Xt + (size_t)j * D);
            proj_full[si] = proj;
            if (proj < ob_thr) {
              lnl++;
              for (int c = 0; c < K; c++) {
                lGL[c] += G[(size_t)j * K + c];
                lHL[c] += H[(size_t)j * K + c];
              }
            }
          }
          tnl[tid] = lnl;
        }
        for (int t2 = 0; t2 < actual_threads; t2++) {
          n_left += tnl[t2];
          for (int c = 0; c < K; c++) {
            GL[c] += tGL[t2][c];
            HL[c] += tHL[t2][c];
          }
        }
      } else
#endif
      {
        for (int si = 0; si < ns; si++) {
          int j = samp[si];
          float proj = sparse_dot_stack(sv, Xt + (size_t)j * D);
          proj_full[si] = proj;
          if (proj < ob_thr) {
            n_left++;
            for (int c = 0; c < K; c++) {
              GL[c] += G[(size_t)j * K + c];
              HL[c] += H[(size_t)j * K + c];
            }
          }
        }
      }
      int n_right = ns - n_left;
      if (n_left < 10 || n_right < 10) return cand_gain[t];
      float HLs = 0.0f, HRs = 0.0f;
      for (int c = 0; c < K; c++) {
        HLs += HL[c];
        HRs += Ht[c] - HL[c];
      }
      if (HLs < MIN_CHILD_W || HRs < MIN_CHILD_W) return cand_gain[t];
      exact_gain = total_base;
      for (int c = 0; c < K; c++) {
        float GR = Gt[c] - GL[c], HR = Ht[c] - HL[c];
        exact_gain += 0.5f * (GL[c] * GL[c] / (HL[c] + reg_lambda + EPS) +
                              GR * GR / (HR + reg_lambda + EPS));
      }
    }

    if (exact_gain > cand_gain[t]) {
      cand_gain[t] = exact_gain;
      cand_thr[t] = ob_thr;
      cand_w[t] = dirs[ob_idx];
      cand_axis[t] = -1;
      cand_proj[t] = std::move(proj_full);
    }
    return cand_gain[t];
  };

  // ── Best-first growth loop ───────────────────────────────────────────────
  node_samp[0].assign(sub, sub + Ns);
  {
    node_hist[0] = get_hist();
    accumulate_hist(sub, Ns, node_hist[0].data(), &node_P[0]);
  }

  std::priority_queue<std::pair<float, int>> frontier;
  if (Ns >= 20) frontier.push({node_P[0], 0});

  int splits_left = max_leaves - 1;
  while (splits_left > 0 && !frontier.empty()) {
    int t_node = frontier.top().second;
    frontier.pop();

    if (node_hist[t_node].empty()) {
      int par_idx = (t_node - 1) / 2;
      int sib = (t_node % 2 == 1) ? t_node + 1 : t_node - 1;
      bool self_small = node_samp[t_node].size() <= node_samp[sib].size();
      int t_small = self_small ? t_node : sib;
      int t_large = self_small ? sib : t_node;

      node_hist[t_small] = get_hist();
      float* GF_RESTRICT hs = node_hist[t_small].data();
      accumulate_hist(node_samp[t_small].data(), (int)node_samp[t_small].size(),
                      hs, nullptr);
      float* GF_RESTRICT hp = node_hist[par_idx].data();
      for (size_t i = 0; i < HSZ; i++) hp[i] -= hs[i];
      node_hist[t_large] = std::move(node_hist[par_idx]);
    }

    if (!oblique_done[t_node]) {
      float ag = eval_axis(t_node);
      float g2 = eval_oblique(t_node);
      oblique_done[t_node] = 1;
      (void)ag;
      if (g2 <= 0.0f) {
        recycle_hist(node_hist[t_node]);
        continue;
      }

      if (!frontier.empty() && g2 < frontier.top().first) {
        frontier.push({g2, t_node});
        continue;
      }
    }
    const auto& samp = node_samp[t_node];
    int tl = 2 * t_node + 1, tr_node = 2 * t_node + 2;
    int depth_t = get_node_depth(t_node);

    if (cand_axis[t_node] < 0) ctx->cache_direction(cand_w[t_node]);
    tree->is_leaf[t_node] = 0;
    tree->split_threshold[t_node] = cand_thr[t_node];
    tree->split_gain[t_node] = cand_gain[t_node];
    std::copy(cand_w[t_node].begin(), cand_w[t_node].end(),
              tree->split_weights.data() + (size_t)t_node * D);
    splits_left--;

    int ax = cand_axis[t_node];
    int bcode = cand_bcode[t_node];
    float thr = cand_thr[t_node];
    const std::vector<float>& proj = cand_proj[t_node];
    std::vector<int> left_sub, right_sub;
    std::vector<float> GL(K, 0.0f), HL(K, 0.0f);
    float PL = 0.0f;
    int ns = (int)samp.size();
#ifdef _OPENMP
    if (ns >= 8192) {
      int nthreads = omp_get_max_threads();
      std::vector<std::vector<int>> tL(nthreads);
      std::vector<std::vector<int>> tR(nthreads);
      std::vector<std::vector<float>> tGL(nthreads, std::vector<float>(K, 0.0f));
      std::vector<std::vector<float>> tHL(nthreads, std::vector<float>(K, 0.0f));
      std::vector<double> tPL(nthreads, 0.0);
      int actual_threads = nthreads;
#pragma omp parallel num_threads(nthreads)
      {
        int tid = omp_get_thread_num();
        if (tid == 0) actual_threads = omp_get_num_threads();
        auto& Lv = tL[tid];
        auto& Rv = tR[tid];
        int chunk_size = (ns + nthreads - 1) / nthreads;
        Lv.reserve(chunk_size);
        Rv.reserve(chunk_size);
        float* GF_RESTRICT gl = tGL[tid].data();
        float* GF_RESTRICT hl = tHL[tid].data();
        double pl = 0.0;
#pragma omp for schedule(static) nowait
        for (int si = 0; si < ns; si++) {
          int j = samp[si];
          bool go_left = (ax >= 0)
                             ? (code[(size_t)j * D + ax] <= (uint8_t)bcode)
                             : (proj[si] < thr);
          if (go_left) {
            Lv.push_back(j);
            const float* GF_RESTRICT gj = G + (size_t)j * K;
            const float* GF_RESTRICT hj = H + (size_t)j * K;
            for (int c = 0; c < K; c++) {
              gl[c] += gj[c];
              hl[c] += hj[c];
              pl += 0.5 * (double)gj[c] * gj[c] /
                    ((double)hj[c] + reg_lambda + EPS);
            }
          } else {
            Rv.push_back(j);
          }
        }
        tPL[tid] = pl;
      }
      size_t nl = 0, nr = 0;
      for (int t = 0; t < actual_threads; t++) {
        nl += tL[t].size();
        nr += tR[t].size();
      }
      left_sub.reserve(nl);
      right_sub.reserve(nr);
      double PLd = 0.0;
      for (int t = 0; t < actual_threads; t++) {
        left_sub.insert(left_sub.end(), tL[t].begin(), tL[t].end());
        right_sub.insert(right_sub.end(), tR[t].begin(), tR[t].end());
        for (int c = 0; c < K; c++) {
          GL[c] += tGL[t][c];
          HL[c] += tHL[t][c];
        }
        PLd += tPL[t];
      }
      PL = (float)PLd;
    } else
#endif
    {
      left_sub.reserve(ns / 2 + 64);
      right_sub.reserve(ns / 2 + 64);
      for (int si = 0; si < ns; si++) {
        int j = samp[si];
        bool go_left = (ax >= 0) ? (code[(size_t)j * D + ax] <= (uint8_t)bcode)
                                 : (proj[si] < thr);
        if (go_left) {
          left_sub.push_back(j);
          const float* GF_RESTRICT gj = G + (size_t)j * K;
          const float* GF_RESTRICT hj = H + (size_t)j * K;
          for (int c = 0; c < K; c++) {
            GL[c] += gj[c];
            HL[c] += hj[c];
            PL += 0.5f * gj[c] * gj[c] / (hj[c] + reg_lambda + EPS);
          }
        } else {
          right_sub.push_back(j);
        }
      }
    }
    node_P[tl] = PL;
    node_P[tr_node] = node_P[t_node] - PL;
    for (int c = 0; c < K; c++) {
      node_G[(size_t)tl * K + c] = GL[c];
      node_H[(size_t)tl * K + c] = HL[c];
      node_G[(size_t)tr_node * K + c] = node_G[(size_t)t_node * K + c] - GL[c];
      node_H[(size_t)tr_node * K + c] = node_H[(size_t)t_node * K + c] - HL[c];
    }
    node_has_tot[tl] = node_has_tot[tr_node] = 1;
    cand_w[t_node].clear();
    cand_proj[t_node].clear();
    cand_proj[t_node].shrink_to_fit();

    node_samp[tl] = std::move(left_sub);
    node_samp[tr_node] = std::move(right_sub);
    bool can_deepen = (depth_t + 1 < internal_depth) && (splits_left > 0);
    if (can_deepen) {
      for (int child : {tl, tr_node}) {
        int cns = (int)node_samp[child].size();
        if (cns < 20) continue;
        if (node_P[child] > 0.0f) frontier.push({node_P[child], child});
      }
    } else {
      recycle_hist(node_hist[t_node]);
    }
  }
  for (int t = 0; t < max_nodes; t++) {
    recycle_hist(node_hist[t]);
  }


  // ── Leaves smoothing ─────────────────────────────────────────────────────
  {
    std::vector<float> sm((size_t)max_nodes * K, 0.0f);
    std::vector<char> hasv(max_nodes, 0);
    for (int t = 0; t < max_nodes; t++) {
      if (!node_has_tot[t] && node_samp[t].empty()) continue;
      if (!node_has_tot[t]) {
        for (int j : node_samp[t]) {
          for (int c = 0; c < K; c++) {
            node_G[(size_t)t * K + c] += G[(size_t)j * K + c];
            node_H[(size_t)t * K + c] += H[(size_t)j * K + c];
          }
        }
      }
      int par_idx = (t - 1) / 2;
      bool use_parent = (t > 0) && hasv[par_idx];
      for (int c = 0; c < K; c++) {
        float Gs = node_G[(size_t)t * K + c], Hs = node_H[(size_t)t * K + c];
        float raw = -Gs / (Hs + reg_lambda + EPS);
        float v = use_parent
                      ? (Hs * raw + reg_lambda * sm[(size_t)par_idx * K + c]) /
                            (Hs + reg_lambda + EPS)
                      : raw;
        sm[(size_t)t * K + c] = v;
        if (tree->is_leaf[t]) tree->leaf_values[(size_t)t * K + c] = v;
      }
      hasv[t] = 1;
    }
  }

  if (out_pred) {
    std::memset(out_pred, 0, (size_t)N * K * sizeof(float));
    std::vector<uint8_t> in_sub(N, 0);
    for (int si = 0; si < Ns; si++) in_sub[sub[si]] = 1;
    for (int t = 0; t < max_nodes; t++) {
      if (!tree->is_leaf[t] || node_samp[t].empty()) continue;
      const float* lv = tree->leaf_values.data() + (size_t)t * K;
      for (int j : node_samp[t]) {
        float* oi = out_pred + (size_t)j * K;
        for (int k = 0; k < K; k++) oi[k] = lv[k];
      }
    }
    if (Ns < N) {
      std::vector<SparseVec> node_nz(max_nodes);
      for (int t = 0; t < max_nodes; t++) {
        if (tree->is_leaf[t]) continue;
        collect_nonzero_stack(tree->split_weights.data() + (size_t)t * D, D,
                              node_nz[t]);
      }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int i = 0; i < N; i++) {
        if (in_sub[i]) continue;
        const float* GF_RESTRICT xi = Xt + (size_t)i * D;
        int t = 0;
        for (int dep = 0; dep < tree->max_depth; dep++) {
          if (tree->is_leaf[t]) break;
          float proj = sparse_dot_stack(node_nz[t], xi);
          t = (proj < tree->split_threshold[t]) ? (2 * t + 1) : (2 * t + 2);
        }
        const float* lv = tree->leaf_values.data() + (size_t)t * K;
        float* oi = out_pred + (size_t)i * K;
        for (int k = 0; k < K; k++) oi[k] = lv[k];
      }
    }
  }
  return static_cast<void*>(tree);
}

// Predict on RAW X
GF_API void gf_predict(void* tree_handle, const float* X, int N, int K,
                             float* out_pred) {
  if (!tree_handle || !out_pred || N <= 0 || !X) return;
  const OQTree* tree = static_cast<const OQTree*>(tree_handle);
  std::memset(out_pred, 0, (size_t)N * K * sizeof(float));
  const int D = tree->D;
  if ((int)tree->na_means.size() != D) {
    _gf_route(tree, X, N, out_pred);
    return;
  }
  const int D_num = tree->D_num;
  std::vector<float> Xt((size_t)N * D);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++) {
    const float* GF_RESTRICT xi = X + (size_t)i * D;
    float* GF_RESTRICT ti = Xt.data() + (size_t)i * D;
    for (int f = 0; f < D; f++) {
      float v = xi[f];
      if (f < D_num) {
        ti[f] = std::isnan(v) ? tree->na_means[f] : v;
      } else {
        int fc = f - D_num;
        if (std::isnan(v) || fc >= (int)tree->cat_ranks.size()) {
          ti[f] = tree->na_means[f];
        } else {
          const auto& m = tree->cat_ranks[fc];
          auto it = m.find((int)std::lrintf(v));
          ti[f] = (it != m.end()) ? it->second : tree->na_means[f];
        }
      }
    }
  }
  _gf_route(tree, Xt.data(), N, out_pred);
}

// Compact routing node: the sparse heap layout (8191 slots for a 64-leaf
// tree) costs ~3 cache misses per visited node; packing the ~127 live nodes
// contiguously keeps the whole tree L1-resident during routing.
struct GFCompactNode {
  SparseVec nz;
  float thr = 0.0f;
  int32_t left = -1, right = -1;  // compact ids; -1 → leaf
};

// Ensemble predict on RAW X: out_pred must be pre-filled with F_init by the
// caller; each tree's leaf values are accumulated scaled by lr.
//
// Layout: trees are processed in tiles of TILE; within a tile ONE parallel
// region walks the rows and routes each row through all tile trees while its
// features sit in L1. vs the old per-tree sweep this divides the N×D memory
// traffic and the fork/join count by TILE (~2× wall on numeric data).
//
// Categorical columns: the per-tree gradient-rank maps share one key set
// (they are refit on the same training categories every round), so raw
// values are hash-looked-up ONCE into dense codes and each tree contributes
// a flat rank table indexed by code — array gathers replace per-tree×per-row
// hash finds (~4× on cat-heavy data).
GF_API void gf_predict_ensemble(void* const* handles, int n_trees,
                                const float* X, int N, int K, float lr,
                                float* out_pred) {
  if (!handles || n_trees <= 0 || !out_pred) return;
  std::vector<const OQTree*> trees;
  trees.reserve(n_trees > 0 ? n_trees : 0);
  for (int t = 0; t < n_trees; t++) {
    const OQTree* tr = static_cast<const OQTree*>(handles[t]);
    if (tr) trees.push_back(tr);
  }
  if (trees.empty()) return;
  const int n_live = (int)trees.size();
  const int D = trees[0]->D;
  const int D_num = trees[0]->D_num;
  const bool has_meta = (int)trees[0]->na_means.size() == D;
  const int D_cat = has_meta ? (D - D_num) : 0;

  // Shared numeric transform (na_means identical across one run's trees).
  std::vector<float> Xt((size_t)N * D);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++) {
    const float* GF_RESTRICT xi = X + (size_t)i * D;
    float* GF_RESTRICT ti = Xt.data() + (size_t)i * D;
    for (int f = 0; f < D; f++) {
      float v = xi[f];
      if (has_meta && f < D_num && std::isnan(v)) v = trees[0]->na_means[f];
      ti[f] = v;
    }
  }

  // ── one-time categorical code assignment (raw id → dense code) ──────────
  // Per column: codes 0..card-1 follow trees[0]'s key set; card is the
  // NaN/unseen slot. codes[i·D_cat+fc] is then a direct index into every
  // tree's rank table.
  std::vector<int32_t> cat_card(D_cat, 0);
  std::vector<int32_t> codes;
  std::vector<std::unordered_map<int, int32_t>> raw2code(D_cat);
  if (D_cat > 0) {
    for (int fc = 0; fc < D_cat; fc++) {
      if (fc >= (int)trees[0]->cat_ranks.size()) continue;
      const auto& m = trees[0]->cat_ranks[fc];
      raw2code[fc].reserve(m.size() * 2);
      int32_t next = 0;
      for (const auto& kv : m) raw2code[fc][kv.first] = next++;
      cat_card[fc] = next;
    }
    codes.assign((size_t)N * D_cat, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < N; i++) {
      const float* GF_RESTRICT xi = X + (size_t)i * D;
      int32_t* GF_RESTRICT ci = codes.data() + (size_t)i * D_cat;
      for (int fc = 0; fc < D_cat; fc++) {
        float v = xi[D_num + fc];
        int32_t c = cat_card[fc];  // NaN / unseen slot
        if (!std::isnan(v)) {
          auto it = raw2code[fc].find((int)std::lrintf(v));
          if (it != raw2code[fc].end()) c = it->second;
        }
        ci[fc] = c;
      }
    }
  }

  static constexpr int TILE = 16;
  std::vector<std::vector<GFCompactNode>> tile_nodes(TILE);
  std::vector<std::vector<float>> tile_leaves(TILE);
  // Per (tile tree, cat col): flat rank table indexed by dense code; the
  // last entry (code == card) is the tree's NaN/unseen fallback.
  std::vector<std::vector<float>> tile_rank(D_cat > 0 ? (size_t)TILE * D_cat
                                                      : 0);
  std::vector<int> heap_of;
  heap_of.reserve(256);

  for (int tile_lo = 0; tile_lo < n_live; tile_lo += TILE) {
    const int tn = std::min(TILE, n_live - tile_lo);

    for (int tt = 0; tt < tn; tt++) {
      const OQTree* tr = trees[tile_lo + tt];

      // ── compact the live subtree (BFS from root) ───────────────────────
      heap_of.clear();
      heap_of.push_back(0);
      for (size_t q = 0; q < heap_of.size(); q++) {
        int h = heap_of[q];
        if (!tr->is_leaf[h]) {
          heap_of.push_back(2 * h + 1);
          heap_of.push_back(2 * h + 2);
        }
      }
      const int M = (int)heap_of.size();
      auto& nodes = tile_nodes[tt];
      auto& leaf_vals = tile_leaves[tt];
      nodes.assign(M, GFCompactNode{});
      leaf_vals.assign((size_t)M * K, 0.0f);
      int next = 1;
      for (int c = 0; c < M; c++) {
        int h = heap_of[c];
        if (tr->is_leaf[h]) {
          const float* lv = tr->leaf_values.data() + (size_t)h * K;
          std::copy(lv, lv + K, leaf_vals.begin() + (size_t)c * K);
        } else {
          collect_nonzero_stack(tr->split_weights.data() + (size_t)h * D, D,
                                nodes[c].nz);
          nodes[c].thr = tr->split_threshold[h];
          nodes[c].left = next;
          nodes[c].right = next + 1;
          next += 2;
        }
      }

      // ── per-tree rank tables (code → encoded value) ─────────────────────
      for (int fc = 0; fc < D_cat; fc++) {
        auto& tbl = tile_rank[(size_t)tt * D_cat + fc];
        tbl.assign((size_t)cat_card[fc] + 1, tr->na_means[D_num + fc]);
        if (fc < (int)tr->cat_ranks.size()) {
          for (const auto& kv : tr->cat_ranks[fc]) {
            auto it = raw2code[fc].find(kv.first);
            if (it != raw2code[fc].end()) tbl[it->second] = kv.second;
          }
        }
      }
    }

    // ── route every row through all tile trees while it is cache-hot ─────
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      std::vector<float> row(D_cat > 0 ? D : 0);
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (int i = 0; i < N; i++) {
        const float* GF_RESTRICT ti = Xt.data() + (size_t)i * D;
        float* GF_RESTRICT oi = out_pred + (size_t)i * K;
        const float* rp = ti;
        if (D_cat > 0) {
          std::memcpy(row.data(), ti, (size_t)D * sizeof(float));
          rp = row.data();
        }
        const int32_t* GF_RESTRICT ci =
            D_cat > 0 ? codes.data() + (size_t)i * D_cat : nullptr;
        for (int tt = 0; tt < tn; tt++) {
          if (D_cat > 0) {
            const auto* tbl0 = tile_rank.data() + (size_t)tt * D_cat;
            for (int fc = 0; fc < D_cat; fc++)
              row[D_num + fc] = tbl0[fc][ci[fc]];
          }
          const GFCompactNode* GF_RESTRICT nd = tile_nodes[tt].data();
          int n = 0;
          while (nd[n].left >= 0) {
            float proj = sparse_dot_stack(nd[n].nz, rp);
            n = (proj < nd[n].thr) ? nd[n].left : nd[n].right;
          }
          const float* lv = tile_leaves[tt].data() + (size_t)n * K;
          for (int k = 0; k < K; k++) oi[k] += lr * lv[k];
        }
      }
    }
  }
}

GF_API void gf_tree_free(void* tree_handle) {
  delete static_cast<OQTree*>(tree_handle);
}

// ─── tree meta (de)serialization ───────────────────────────────────────────
GF_API void gf_tree_meta_sizes(void* tree_handle, int* sizes) {
  const OQTree* tree = static_cast<const OQTree*>(tree_handle);
  sizes[0] = tree->D_num;
  sizes[1] = (int)tree->cat_ranks.size();
  int total = 0;
  for (const auto& m : tree->cat_ranks) total += (int)m.size();
  sizes[2] = total;
  sizes[3] = (int)tree->na_means.size();
}

GF_API void gf_tree_export_meta(void* tree_handle, float* na_means,
                                      int* cat_sizes, int* cat_keys,
                                      float* cat_vals) {
  const OQTree* tree = static_cast<const OQTree*>(tree_handle);
  for (size_t i = 0; i < tree->na_means.size(); i++)
    na_means[i] = tree->na_means[i];
  int off = 0;
  for (size_t fc = 0; fc < tree->cat_ranks.size(); fc++) {
    const auto& m = tree->cat_ranks[fc];
    cat_sizes[fc] = (int)m.size();
    for (const auto& kv : m) {
      cat_keys[off] = kv.first;
      cat_vals[off] = kv.second;
      off++;
    }
  }
}

GF_API void gf_tree_import_meta(void* tree_handle, int D_num,
                                      const float* na_means, int na_len,
                                      const int* cat_sizes, int D_cat,
                                      const int* cat_keys,
                                      const float* cat_vals) {
  OQTree* tree = static_cast<OQTree*>(tree_handle);
  tree->D_num = D_num;
  tree->na_means.assign(na_means, na_means + na_len);
  tree->cat_ranks.assign(D_cat, {});
  int off = 0;
  for (int fc = 0; fc < D_cat; fc++) {
    auto& m = tree->cat_ranks[fc];
    m.reserve((size_t)cat_sizes[fc] * 2);
    for (int e = 0; e < cat_sizes[fc]; e++) {
      m[cat_keys[off]] = cat_vals[off];
      off++;
    }
  }
}

// ─── tree structure (de)serialization — ported from the removed bfstree.cpp ─
GF_API int oqtree_get_K(void* handle) { return static_cast<OQTree*>(handle)->K; }
GF_API int oqtree_get_max_depth(void* handle) {
  return static_cast<OQTree*>(handle)->max_depth;
}
GF_API int oqtree_get_total_nodes(void* handle) {
  return static_cast<OQTree*>(handle)->total_nodes;
}
GF_API int oqtree_get_D(void* handle) {
  return static_cast<OQTree*>(handle)->D;
}

GF_API void oqtree_export(void* handle, int* split_hyp_idx,
                          float* split_threshold, float* leaf_values,
                          uint8_t* is_leaf) {
  const OQTree* tree = static_cast<const OQTree*>(handle);
  int n = tree->total_nodes, K = tree->K;
  for (int i = 0; i < n; ++i) split_hyp_idx[i] = tree->split_hyp_idx[i];
  for (int i = 0; i < n; ++i) split_threshold[i] = tree->split_threshold[i];
  for (size_t i = 0; i < (size_t)n * K; ++i) leaf_values[i] = tree->leaf_values[i];
  for (int i = 0; i < n; ++i) is_leaf[i] = tree->is_leaf[i];
}

GF_API void* oqtree_from_arrays(const int* split_hyp_idx,
                                const float* split_threshold,
                                const float* leaf_values,
                                const uint8_t* is_leaf, int total_nodes, int K,
                                int max_depth) {
  OQTree* tree = new OQTree();
  tree->total_nodes = total_nodes;
  tree->K = K;
  tree->max_depth = max_depth;
  tree->split_hyp_idx.assign(split_hyp_idx, split_hyp_idx + total_nodes);
  tree->split_threshold.assign(split_threshold, split_threshold + total_nodes);
  tree->leaf_values.assign(leaf_values, leaf_values + (size_t)total_nodes * K);
  tree->is_leaf.assign(is_leaf, is_leaf + total_nodes);
  return static_cast<void*>(tree);
}

GF_API void oqtree_get_split_weights(void* handle, float* out_weights) {
  const OQTree* tree = static_cast<const OQTree*>(handle);
  if (!tree->split_weights.empty()) {
    std::memcpy(out_weights, tree->split_weights.data(),
                tree->split_weights.size() * sizeof(float));
  }
}

GF_API void oqtree_set_split_weights(void* handle, int D, const float* weights) {
  OQTree* tree = static_cast<OQTree*>(handle);
  tree->D = D;
  tree->split_weights.assign(weights, weights + (size_t)tree->total_nodes * D);
}

GF_API void gf_update_gradients(const float* F, const float* oh, int N, int K, float* G, float* H) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++) {
    size_t offset = (size_t)i * K;

    // Find max F for numerical stability
    float fmax = F[offset];
    for (int c = 1; c < K; c++) {
      if (F[offset + c] > fmax) {
        fmax = F[offset + c];
      }
    }

    // Sum of exponentials
    double sum_exp = 0.0;
    for (int c = 0; c < K; c++) {
      sum_exp += std::exp(F[offset + c] - fmax);
    }

    // Compute P, G, H
    for (int c = 0; c < K; c++) {
      float p = (float)(std::exp(F[offset + c] - fmax) / sum_exp);
      G[offset + c] = p - oh[offset + c];
      H[offset + c] = p * (1.0f - p);
    }
  }
}

}  // extern "C"