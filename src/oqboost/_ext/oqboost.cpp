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

static inline double thr_l1(double g, float alpha) {
  if (alpha <= 0.0f) return g;
  if (g > alpha) return g - alpha;
  if (g < -alpha) return g + alpha;
  return 0.0;
}

static constexpr int OBLIQUE_MIN = 64;  // min node size to fit directions

// ─── Binning context (per dataset, reused across all boosting rounds) ───────
struct OQBoostCtx {
  int N = 0, D = 0, D_num = 0, D_cat = 0;
  int n_bins = 255;
  std::vector<uint8_t> code;            // N·D uint8 bin codes
  std::vector<float> ax_min, ax_range;  // per-feature bin frame
  std::vector<float> Ximp;              // N·D transformed x̃ (numeric static,
                                        // cat columns rewritten per round)
  std::vector<float> col_mean;          // [D_num] numeric impute means μ_f
  std::vector<uint8_t> is_nan;          // N·D missing value mask (1 if NaN, 0 if not)

  // Categorical raw-value dictionaries (static across rounds).
  std::vector<std::unordered_map<int, int>> cat_id;  // raw → dense id
  std::vector<int> cat_card;       // per cat col: n_distinct + 1 (NaN slot)
  std::vector<int32_t> cat_dense;  // N·D_cat dense ids (NaN → card-1)

  // Persistent direction cache
  static constexpr int DIR_CACHE_MAX = 32;
  std::vector<std::vector<float>> dir_cache;  // dense D each
  int dir_cache_next = 0;

  // Workspaces for oblique search (pre-allocated to avoid frequent malloc)
  std::vector<float> Xe_workspace; // Size: ne * D
  std::vector<float> Pj_workspace; // Size: n_dirs * ne

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

  // ── Oblique scratch buffers (pre-allocated per context) ───────────────────
  // Flat direction matrix: dirs_buf[i*D + d] = component d of direction i.
  // dirs_n is reset to 0 at the start of each eval_oblique call.
  static constexpr int DIRS_MAX = 160;  // pool(64)+pobs(8)+cache(32)+margin
  std::vector<float> dirs_buf;       // capacity: DIRS_MAX * D
  int dirs_n = 0;                    // active directions count
  std::vector<float> scratch_w;      // D floats: temp buffer for one direction
  std::vector<int>   samp_e_buf;     // EST_NE_MAX ints: stride_cap result
  std::vector<float> scratch_dir_gain;   // DIRS_MAX floats
  std::vector<float> scratch_dir_thr;    // DIRS_MAX floats
  std::vector<int>   scratch_cand_feats; // D ints: strategy B candidate feats
  std::vector<float> scratch_cand_probs; // D floats: strategy B candidate probs
  std::vector<float> scratch_Gt;
  std::vector<float> scratch_Ht;
  std::vector<float> scratch_eGt;
  std::vector<float> scratch_eHt;
  std::vector<float> scratch_cg_s;
  std::vector<float> scratch_add_s;
  std::vector<float> scratch_fscore;
  std::vector<float> scratch_prob;
};

static inline int get_node_depth(int t) {
  // Heap index depth = floor(log2(t+1)). Single-cycle bit instruction.
#if defined(__GNUC__) || defined(__clang__)
  return (t == 0) ? 0 : (31 - __builtin_clz((unsigned)(t + 1)));
#elif defined(_MSC_VER)
  unsigned long idx = 0;
  _BitScanReverse(&idx, (unsigned)(t + 1));
  return (t == 0) ? 0 : (int)idx;
#else
  int depth = 0;
  while (t > 0) { t = (t - 1) / 2; depth++; }
  return depth;
#endif
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

// Route a TRANSFORMED matrix x̃ (can have NaNs, cats already rank-encoded).
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
      bool has_nan = false;
      const SparseVec& sv = node_nz[t];
      if (sv.size == 1) {
        int idx = sv.indices[0];
        if (idx < tree->D_num && std::isnan(xi[idx])) {
          has_nan = true;
        }
      }
      int next_t;
      if (has_nan) {
        next_t = (tree->default_left[t] != 0) ? (2 * t + 1) : (2 * t + 2);
      } else {
        float proj = 0.0f;
        if (sv.size == 1) {
          proj = sv.values[0] * xi[sv.indices[0]];
        } else {
          for (int k = 0; k < sv.size; k++) {
            float v = xi[sv.indices[k]];
            if (std::isnan(v)) {
              v = tree->na_means[sv.indices[k]];
            }
            proj += sv.values[k] * v;
          }
        }
        next_t = (proj < tree->split_threshold[t]) ? (2 * t + 1) : (2 * t + 2);
      }
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
                                 const int* sub, int Ns, int max_bin) {
  auto* ctx = new OQBoostCtx();
  ctx->n_bins = std::max(2, std::min(max_bin, 255));
  const int AX_BINS = ctx->n_bins;
  ctx->N = N;
  ctx->D = D;
  ctx->D_num = D_num;
  ctx->D_cat = D - D_num;
  ctx->ax_min.assign(D, 0.0f);
  ctx->ax_range.assign(D, 0.0f);
  ctx->col_mean.assign(D_num, 0.0f);
  ctx->Ximp.assign((size_t)N * D, 0.0f);
  ctx->code.assign((size_t)N * D, 0);
  ctx->is_nan.assign((size_t)N * D, 0);

  // Pre-allocate oblique scratch buffers (D known here).
  ctx->dirs_buf.resize((size_t)OQBoostCtx::DIRS_MAX * D, 0.0f);
  ctx->scratch_w.resize(D, 0.0f);
  ctx->samp_e_buf.reserve(EST_NE_MAX);
  ctx->scratch_dir_gain.reserve(OQBoostCtx::DIRS_MAX);
  ctx->scratch_dir_thr.reserve(OQBoostCtx::DIRS_MAX);
  ctx->scratch_cand_feats.reserve(D);
  ctx->scratch_cand_probs.reserve(D);
  ctx->scratch_cg_s.resize(D, 0.0f);
  ctx->scratch_add_s.resize(D, 0.0f);
  ctx->scratch_fscore.resize(D, 0.0f);
  ctx->scratch_prob.resize(D, 0.0f);
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
      ax_scale[f] = (float)(AX_BINS - 1) / (range + EPS);
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
      bool is_missing = std::isnan(v);
      if (is_missing) {
        ctx->is_nan[(size_t)i * D + f] = 1;
        v = ctx->col_mean[f];
      }
      ti[f] = v;
      if (ctx->ax_range[f] == 0.0f) continue;
      int b;
      if (is_missing) {
        b = AX_BINS - 1;
      } else {
        b = (int)((v - ctx->ax_min[f]) * ax_scale[f]);
        if (b < 0) b = 0;
        if (b >= AX_BINS - 1) b = AX_BINS - 2;
      }
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
          ctx->is_nan[(size_t)i * D + f] = 1;
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

static inline bool has_nan_in_split(const uint8_t* is_nan, int i, int D, const SparseVec& sv) {
  if (sv.size == 1) {
    return is_nan[(size_t)i * D + sv.indices[0]] != 0;
  }
  return false;
}

static inline float compute_split_score(const std::vector<double>& GL, const std::vector<double>& HL,
                                        const std::vector<double>& GR, const std::vector<double>& HR,
                                        int K, float reg_lambda) {
  float score = 0.0f;
  for (int c = 0; c < K; c++) {
    score += 0.5f * (float)(GL[c] * GL[c] / (HL[c] + reg_lambda + EPS) +
                            GR[c] * GR[c] / (HR[c] + reg_lambda + EPS));
  }
  return score;
}

// ─── gf_build — one boosting round on a binning context ────────────────
GF_API void* gf_build(void* ctx_handle, const float* X, const float* G,
                               const float* H, int K, const int* sub, int Ns,
                               int max_depth, float reg_lambda,
                               float inherited_rp_ratio, float mutation_rate,
                               float mutation_strength, int seed, int pobs,
                               float reg_alpha, float gamma, float min_child_weight,
                               float colsample_bynode, int max_leaves, float* out_pred) {

  (void)X;
  auto* ctx = static_cast<OQBoostCtx*>(ctx_handle);
  const int AX_BINS = ctx->n_bins;
  const int D = ctx->D, D_num = ctx->D_num, D_cat = ctx->D_cat, N = ctx->N;
  ctx->scratch_Gt.resize(K);
  ctx->scratch_Ht.resize(K);
  ctx->scratch_eGt.resize(K);
  ctx->scratch_eHt.resize(K);
  const float* GF_RESTRICT Xt = ctx->Ximp.data();
  const int STRIDE = 2 * K + 1;
  const size_t HSZ = (size_t)D * AX_BINS * STRIDE;



  const int internal_depth = std::min(2 * max_depth, 22);
  int max_nodes = (1 << (internal_depth + 1)) - 1;
  auto* tree = new OQTree();
  tree->K = K;
  tree->D = D;
  tree->D_num = D_num;
  tree->max_depth = internal_depth;
  tree->total_nodes = max_nodes;
  tree->is_leaf.assign(max_nodes, 1);
  tree->default_left.assign(max_nodes, 1);  // default missing values to Left child
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
    int nthreads = 1;
#ifdef _OPENMP
    if (nr >= 2048) {
      nthreads = omp_get_max_threads();
    } else if (nr >= 512) {
      nthreads = std::min(4, omp_get_max_threads());
    } else if (nr >= 128) {
      nthreads = std::min(2, omp_get_max_threads());
    }
#endif

    if (nthreads > 1) {
      if (D >= nthreads && D >= 2) {
        // Block-wise Feature-parallelism (highly efficient for larger D, zero merge overhead)
        int block_size = std::max(1, D / nthreads);
#pragma omp parallel for schedule(static) num_threads(nthreads)
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
        // Flat Thread-local Sample-parallelism (extremely fast for smaller D)
        std::vector<float> local_hists((size_t)nthreads * HSZ, 0.0f);
#pragma omp parallel num_threads(nthreads)
        {
          int tid = omp_get_thread_num();
          float* GF_RESTRICT lb = local_hists.data() + (size_t)tid * HSZ;

#pragma omp for schedule(static)
          for (int si = 0; si < nr; si++) {
            int j = rows[si];
            const uint8_t* GF_RESTRICT cj = code + (size_t)j * D;
            const float* GF_RESTRICT gj = G + (size_t)j * K;
            const float* GF_RESTRICT hj = H + (size_t)j * K;
            for (int f = 0; f < D; f++) {
              int b = cj[f];
              float* GF_RESTRICT slot = lb + ((size_t)f * AX_BINS + b) * STRIDE;
              for (int c = 0; c < K; c++) {
                slot[c] += gj[c];
                slot[K + c] += hj[c];
              }
              slot[2 * K] += 1.0f;
            }
          }

#pragma omp for schedule(static)
          for (size_t i = 0; i < HSZ; i++) {
            float s = 0.0f;
            for (int t = 0; t < nthreads; t++) {
              s += local_hists[(size_t)t * HSZ + i];
            }
            hb[i] = s;
          }
        }
      }

      if (node_P_out) {
        double P_sum = 0.0;
#pragma omp parallel for reduction(+:P_sum) schedule(static) num_threads(nthreads)
        for (int si = 0; si < nr; si++) {
          int j = rows[si];
          const float* GF_RESTRICT gj = G + (size_t)j * K;
          const float* GF_RESTRICT hj = H + (size_t)j * K;
          for (int c = 0; c < K; c++) {
            P_sum += 0.5 * (double)gj[c] * gj[c] / ((double)hj[c] + reg_lambda + EPS);
          }
        }
        *node_P_out = (float)P_sum;
      }
    } else {
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
    }
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
  std::vector<uint8_t> cand_default_left(max_nodes, 1);
  std::vector<std::vector<float>> cand_w(max_nodes);
  std::vector<std::vector<float>> cand_proj(max_nodes);

  std::vector<char> oblique_done(max_nodes, 0);

  // ── Phase 1: Axis scan ───────────────────────────────────────────────────
  // feat_mask is now generated once per node in the growth loop and shared.
  auto eval_axis = [&](int t, const std::vector<char>& feat_mask) -> float {
    int ns = (int)node_samp[t].size();
    const float* GF_RESTRICT hb = node_hist[t].data();

    auto& Gt = ctx->scratch_Gt;
    auto& Ht = ctx->scratch_Ht;
    std::fill(Gt.begin(), Gt.end(), 0.0f);
    std::fill(Ht.begin(), Ht.end(), 0.0f);
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
    for (int c = 0; c < K; c++) {
      double gt_l1 = thr_l1(Gt[c], reg_alpha);
      total_base -= 0.5f * gt_l1 * gt_l1 / (Ht[c] + reg_lambda + EPS);
    }

    float best_gain = 0.0f, best_thr = 0.0f;
    int best_axis = -1, best_axis_b = 0;
    uint8_t best_default_left = 1;

    {
      float l_gain = 0.0f, l_thr = 0.0f;
      int l_axis = -1, l_b = 0;
      uint8_t l_default_left = 1;
#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        float thread_best_gain = 0.0f, thread_best_thr = 0.0f;
        int thread_best_axis = -1, thread_best_b = 0;
        uint8_t thread_best_default_left = 1;
        
        constexpr int K_MAX_STACK = 64;
        std::vector<float> thread_Gc_fb, thread_Hc_fb;
        std::vector<double> bG_nan_fb, bH_nan_fb, bG_clean_fb, bH_clean_fb;
        std::vector<double> bGc_fb, bHc_fb, bGL_fb, bHL_fb, bGR_fb, bHR_fb;

        float thread_Gc_stack[K_MAX_STACK];
        float thread_Hc_stack[K_MAX_STACK];
        double bG_nan_stack[K_MAX_STACK];
        double bH_nan_stack[K_MAX_STACK];
        double bG_clean_stack[K_MAX_STACK];
        double bH_clean_stack[K_MAX_STACK];
        double bGc_stack[K_MAX_STACK];
        double bHc_stack[K_MAX_STACK];
        double bGL_stack[K_MAX_STACK];
        double bHL_stack[K_MAX_STACK];
        double bGR_stack[K_MAX_STACK];
        double bHR_stack[K_MAX_STACK];

        float* GF_RESTRICT thread_Gc = thread_Gc_stack;
        float* GF_RESTRICT thread_Hc = thread_Hc_stack;
        double* GF_RESTRICT bG_nan = bG_nan_stack;
        double* GF_RESTRICT bH_nan = bH_nan_stack;
        double* GF_RESTRICT bG_clean = bG_clean_stack;
        double* GF_RESTRICT bH_clean = bH_clean_stack;
        double* GF_RESTRICT bGc = bGc_stack;
        double* GF_RESTRICT bHc = bHc_stack;
        double* GF_RESTRICT bGL = bGL_stack;
        double* GF_RESTRICT bHL = bHL_stack;
        double* GF_RESTRICT bGR = bGR_stack;
        double* GF_RESTRICT bHR = bHR_stack;

        if (K > K_MAX_STACK) {
          thread_Gc_fb.resize(K); thread_Hc_fb.resize(K);
          bG_nan_fb.resize(K); bH_nan_fb.resize(K);
          bG_clean_fb.resize(K); bH_clean_fb.resize(K);
          bGc_fb.resize(K); bHc_fb.resize(K);
          bGL_fb.resize(K); bHL_fb.resize(K);
          bGR_fb.resize(K); bHR_fb.resize(K);

          thread_Gc = thread_Gc_fb.data();
          thread_Hc = thread_Hc_fb.data();
          bG_nan = bG_nan_fb.data();
          bH_nan = bH_nan_fb.data();
          bG_clean = bG_clean_fb.data();
          bH_clean = bH_clean_fb.data();
          bGc = bGc_fb.data();
          bHc = bHc_fb.data();
          bGL = bGL_fb.data();
          bHL = bHL_fb.data();
          bGR = bGR_fb.data();
          bHR = bHR_fb.data();
        }
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
        for (int f = 0; f < D; f++) {
          if (!feat_mask[f]) continue;
          if (ctx->ax_range[f] == 0.0f) continue;
          const float* GF_RESTRICT fbuf = hb + (size_t)f * AX_BINS * STRIDE;
          if (f < D_num) {
            const float* GF_RESTRICT nan_slot = fbuf + (size_t)(AX_BINS - 1) * STRIDE;
            int n_nan = (int)nan_slot[2 * K];
            for (int c = 0; c < K; c++) {
              bG_nan[c] = nan_slot[c];
              bH_nan[c] = nan_slot[K + c];
            }
            int n_clean_tot = ns - n_nan;
            for (int c = 0; c < K; c++) {
              bG_clean[c] = Gt[c] - bG_nan[c];
              bH_clean[c] = Ht[c] - bH_nan[c];
            }
            std::fill_n(bGc, K, 0.0);
            std::fill_n(bHc, K, 0.0);
            int n_left = 0;
            for (int b = 0; b < AX_BINS - 1; b++) {
              const float* GF_RESTRICT slot = fbuf + (size_t)b * STRIDE;
              n_left += (int)slot[2 * K];
              for (int c = 0; c < K; c++) {
                bGc[c] += slot[c];
                bHc[c] += slot[K + c];
              }
              int n_clean_R = n_clean_tot - n_left;
              // Option L: NaNs go Left
              {
                int n_L = n_left + n_nan;
                int n_R = n_clean_R;
                if (n_L >= 10 && n_R >= 10) {
                  double H_L_sum = 0.0, H_R_sum = 0.0;
                  double* GF_RESTRICT GL = bGL;
                  double* GF_RESTRICT HL = bHL;
                  double* GF_RESTRICT GR = bGR;
                  double* GF_RESTRICT HR = bHR;
                  for (int c = 0; c < K; c++) {
                    GL[c] = bGc[c] + bG_nan[c];
                    HL[c] = bHc[c] + bH_nan[c];
                    GR[c] = bG_clean[c] - bGc[c];
                    HR[c] = bH_clean[c] - bHc[c];
                    H_L_sum += HL[c];
                    H_R_sum += HR[c];
                  }
                  if (H_L_sum >= min_child_weight && H_R_sum >= min_child_weight) {
                    float gain = total_base;
                    for (int c = 0; c < K; c++) {
                      double gl_l1 = thr_l1(GL[c], reg_alpha);
                      double gr_l1 = thr_l1(GR[c], reg_alpha);
                      gain += 0.5f * (float)(gl_l1 * gl_l1 / (HL[c] + reg_lambda + EPS) +
                                             gr_l1 * gr_l1 / (HR[c] + reg_lambda + EPS));
                    }
                    if (gain > thread_best_gain || (gain == thread_best_gain && thread_best_axis >= 0 && f < thread_best_axis)) {
                      thread_best_gain = gain;
                      thread_best_axis = f;
                      thread_best_b = b;
                      thread_best_thr = ctx->ax_min[f] + ((float)(b + 1) / AX_BINS) * ctx->ax_range[f];
                      thread_best_default_left = 1;
                    }
                  }
                }
              }
              // Option R: NaNs go Right
              {
                int n_L = n_left;
                int n_R = n_clean_R + n_nan;
                if (n_L >= 10 && n_R >= 10) {
                  double H_L_sum = 0.0, H_R_sum = 0.0;
                  double* GF_RESTRICT GL = bGL;
                  double* GF_RESTRICT HL = bHL;
                  double* GF_RESTRICT GR = bGR;
                  double* GF_RESTRICT HR = bHR;
                  for (int c = 0; c < K; c++) {
                    GL[c] = bGc[c];
                    HL[c] = bHc[c];
                    GR[c] = bG_clean[c] - bGc[c] + bG_nan[c];
                    HR[c] = bH_clean[c] - bHc[c] + bH_nan[c];
                    H_L_sum += HL[c];
                    H_R_sum += HR[c];
                  }
                  if (H_L_sum >= min_child_weight && H_R_sum >= min_child_weight) {
                    float gain = total_base;
                    for (int c = 0; c < K; c++) {
                      double gl_l1 = thr_l1(GL[c], reg_alpha);
                      double gr_l1 = thr_l1(GR[c], reg_alpha);
                      gain += 0.5f * (float)(gl_l1 * gl_l1 / (HL[c] + reg_lambda + EPS) +
                                             gr_l1 * gr_l1 / (HR[c] + reg_lambda + EPS));
                    }
                    if (gain > thread_best_gain || (gain == thread_best_gain && thread_best_axis >= 0 && f < thread_best_axis)) {
                      thread_best_gain = gain;
                      thread_best_axis = f;
                      thread_best_b = b;
                      thread_best_thr = ctx->ax_min[f] + ((float)(b + 1) / AX_BINS) * ctx->ax_range[f];
                      thread_best_default_left = 0;
                    }
                  }
                }
              }
            }
          } else {
            // Categorical: normal scan, NaNs are handled natively by category codes (default_left = 1)
            std::fill_n(thread_Gc, K, 0.0f);
            std::fill_n(thread_Hc, K, 0.0f);
            int n_left = 0;
            for (int b = 0; b < AX_BINS - 1; b++) {
              const float* GF_RESTRICT slot = fbuf + (size_t)b * STRIDE;
              n_left += (int)slot[2 * K];
              for (int c = 0; c < K; c++) {
                thread_Gc[c] += slot[c];
                thread_Hc[c] += slot[K + c];
              }
              int n_right = ns - n_left;
              if (n_left < 10 || n_right < 10) continue;
              float Hcs = 0.0f, Hrs = 0.0f;
              for (int c = 0; c < K; c++) {
                Hcs += thread_Hc[c];
                Hrs += Ht[c] - thread_Hc[c];
              }
              if (Hcs < min_child_weight || Hrs < min_child_weight) continue;
              float gain = total_base;
              for (int c = 0; c < K; c++) {
                float Gr = Gt[c] - thread_Gc[c], Hr = Ht[c] - thread_Hc[c];
                double tc_l1 = thr_l1(thread_Gc[c], reg_alpha);
                double gr_l1 = thr_l1(Gr, reg_alpha);
                gain += 0.5f * (tc_l1 * tc_l1 / (thread_Hc[c] + reg_lambda + EPS) +
                                gr_l1 * gr_l1 / (Hr + reg_lambda + EPS));
              }
              if (gain > thread_best_gain || (gain == thread_best_gain && thread_best_axis >= 0 && f < thread_best_axis)) {
                thread_best_gain = gain;
                thread_best_axis = f;
                thread_best_b = b;
                thread_best_thr = ctx->ax_min[f] + ((float)(b + 1) / AX_BINS) * ctx->ax_range[f];
                thread_best_default_left = 1;
              }
            }
          }
        }
#ifdef _OPENMP
#pragma omp critical
#endif
        {
          if (thread_best_axis >= 0 &&
              (thread_best_gain > l_gain ||
               (thread_best_gain == l_gain && (l_axis < 0 || thread_best_axis < l_axis)))) {
            l_gain = thread_best_gain;
            l_axis = thread_best_axis;
            l_b = thread_best_b;
            l_thr = thread_best_thr;
            l_default_left = thread_best_default_left;
          }
        }
      }
      best_gain = l_gain;
      best_axis = l_axis;
      best_axis_b = l_b;
      best_thr = l_thr;
      best_default_left = l_default_left;
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
  auto eval_oblique = [&](int t, const std::vector<char>& feat_mask) -> float {
    const auto& samp = node_samp[t];
    int ns = (int)samp.size();
    if (ns < OBLIQUE_MIN) return cand_gain[t];

    auto& Gt = ctx->scratch_Gt;
    auto& Ht = ctx->scratch_Ht;
    for (int c = 0; c < K; c++) {
      Gt[c] = node_G[(size_t)t * K + c];
      Ht[c] = node_H[(size_t)t * K + c];
    }
    float total_base = 0.0f;
    for (int c = 0; c < K; c++) {
      double gt_l1 = thr_l1(Gt[c], reg_alpha);
      total_base -= 0.5f * gt_l1 * gt_l1 / (Ht[c] + reg_lambda + EPS);
    }

    // Use ctx pre-allocated scratch instead of allocating per eval_oblique call.
    auto& samp_e = ctx->samp_e_buf;
    samp_e.clear();
    if ((int)samp.size() <= EST_NE_MAX) {
      samp_e.assign(samp.begin(), samp.end());
    } else {
      samp_e.resize(EST_NE_MAX);
      double step = (double)samp.size() / EST_NE_MAX;
      for (int i = 0; i < EST_NE_MAX; i++)
        samp_e[i] = samp[(size_t)(i * step)];
    }
    int ne = (int)samp_e.size();
    bool exact_eval = (ne == ns);
    auto& eGt = ctx->scratch_eGt;
    auto& eHt = ctx->scratch_eHt;
    std::fill(eGt.begin(), eGt.end(), 0.0f);
    std::fill(eHt.begin(), eHt.end(), 0.0f);
    if (exact_eval) {
      std::copy(Gt.begin(), Gt.end(), eGt.begin());
      std::copy(Ht.begin(), Ht.end(), eHt.begin());
    } else {
      for (int j : samp_e) {
        for (int c = 0; c < K; c++) {
          eGt[c] += G[(size_t)j * K + c];
          eHt[c] += H[(size_t)j * K + c];
        }
      }
    }
    float e_base = 0.0f;
    for (int c = 0; c < K; c++) {
      double egt_l1 = thr_l1(eGt[c], reg_alpha);
      e_base -= 0.5f * egt_l1 * egt_l1 / (eHt[c] + reg_lambda + EPS);
    }

    int kdom = dominant_class(samp_e, G, K);

    // Gather the evaluation panel contiguously once: every later pass (SIS
    // accumulation, all candidate projections) then walks Xe sequentially
    // instead of striding through Xt rows scattered across the full matrix.
    ctx->Xe_workspace.resize((size_t)ne * D);
    float* Xe_ptr = ctx->Xe_workspace.data();
    for (int i = 0; i < ne; i++) {
      std::memcpy(Xe_ptr + (size_t)i * D, Xt + (size_t)samp_e[i] * D,
                  (size_t)D * sizeof(float));
    }

    auto& cg_s = ctx->scratch_cg_s;
    auto& add_s = ctx->scratch_add_s;
    std::fill(cg_s.begin(), cg_s.end(), 0.0f);
    std::fill(add_s.begin(), add_s.end(), 0.0f);
    for (int i = 0; i < ne; i++) {
      const float* GF_RESTRICT xi = Xe_ptr + (size_t)i * D;
      int idx = samp_e[i];
      float gi = G[(size_t)idx * K + kdom];
      float hi = H[(size_t)idx * K + kdom];
      for (int d = 0; d < D; d++) {
        cg_s[d] += xi[d] * gi;
        add_s[d] += hi * xi[d] * xi[d];
      }
    }
    auto& fscore = ctx->scratch_fscore;
    for (int d = 0; d < D; d++) {
      fscore[d] = std::abs(cg_s[d]) / std::sqrt(add_s[d] + reg_lambda + EPS);
    }

    // Use ctx flat buffer for direction candidates — eliminates n_dirs heap
    // mallocs per call. dirs_flat[i*D .. (i+1)*D) holds direction vector i.
    int& dirs_n = ctx->dirs_n;
    dirs_n = 0;
    float* GF_RESTRICT dirs_flat = ctx->dirs_buf.data();
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

    auto& prob = ctx->scratch_prob;
    for (int d = 0; d < D; d++) {
                    prob[d] = feat_mask[d] ? (fscore[d] + 1e-6f) : 0.0f;
    }
    std::discrete_distribution<int> feat_dist(prob.begin(), prob.end());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    auto fill_pobs = [&](int count) {
      if (count <= 0) return;
      int m = (int)std::lround(std::sqrt((float)D));
      m = std::max(2, std::min({m, D_SUB_MAX, D}));
      std::normal_distribution<float> gauss(0.0f, 1.0f);
      int target = dirs_n + count;
      while (dirs_n < target && dirs_n < OQBoostCtx::DIRS_MAX) {
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
        float B_flat[D_SUB_MAX * D_SUB_MAX];
        for (int ii = 0; ii < mm * mm; ii++) B_flat[ii] = gauss(rng);
        int emitted = 0;
        for (int i = 0; i < mm && dirs_n < target && dirs_n < OQBoostCtx::DIRS_MAX; i++) {
          for (int j = 0; j < i; j++) {
            float dot = 0.0f;
            for (int d2 = 0; d2 < mm; d2++) dot += B_flat[i*mm+d2] * B_flat[j*mm+d2];
            for (int d2 = 0; d2 < mm; d2++) B_flat[i*mm+d2] -= dot * B_flat[j*mm+d2];
          }
          float n2 = 0.0f;
          for (int d2 = 0; d2 < mm; d2++) n2 += B_flat[i*mm+d2] * B_flat[i*mm+d2];
          if (n2 < 1e-12f) continue;
          float inv = 1.0f / std::sqrt(n2);
          float* GF_RESTRICT slot = dirs_flat + (size_t)dirs_n * D;
          std::fill(slot, slot + D, 0.0f);
          for (int d2 = 0; d2 < mm; d2++) slot[S[d2]] = B_flat[i*mm+d2] * inv;
          dirs_n++;
          fam.push_back('p');
          emitted++;
        }
        if (emitted == 0) break;
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

    int depth_t = get_node_depth(t);
    int pool_budget = (depth_t <= 2) ? 64 : 8;

    int n_inherited = (int)std::round((float)pool_budget * inherited_rp_ratio);
    if (n_inherited < 0) n_inherited = 0;
    if (n_inherited > pool_budget) n_inherited = pool_budget;
    int n_global = pool_budget - n_inherited;

    int n_pobs_extra = pobs ? std::min(8, n_inherited) : 0;
    n_inherited -= n_pobs_extra;

    if (has_parent && parent_nz.size > 0) {
      float local_mutation_rate =
          mutation_rate / std::sqrt(1.0f + (float)depth_t);
      float local_mutation_strength =
          mutation_strength / (1.0f + (float)depth_t);

      for (int r = 0; r < n_inherited; r++) {
        float strategy_draw = dist(rng);
        bool do_strategy_a = false;
        bool do_strategy_b = false;
        bool do_strategy_c = false;

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

        float* GF_RESTRICT w_rand = ctx->scratch_w.data();
        std::fill(w_rand, w_rand + D, 0.0f);

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
            auto& candidate_feats = ctx->scratch_cand_feats;
            auto& candidate_probs = ctx->scratch_cand_probs;
            candidate_feats.clear();
            candidate_probs.clear();
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
        if (dirs_n < OQBoostCtx::DIRS_MAX) {
          float* GF_RESTRICT slot = dirs_flat + (size_t)dirs_n * D;
          if (norm > 1e-12f) {
            float inv_norm = 1.0f / norm;
            for (int d = 0; d < D; d++) slot[d] = w_rand[d] * inv_norm;
          } else {
            // Fallback: keep parent's first nonzero direction.
            std::fill(slot, slot + D, 0.0f);
            slot[parent_nz.indices[0]] = parent_nz.values[0];
          }
          dirs_n++;
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
      if (dirs_n < OQBoostCtx::DIRS_MAX) {
        std::memcpy(dirs_flat + (size_t)dirs_n * D, cw.data(),
                    (size_t)D * sizeof(float));
        dirs_n++;
      }
      fam.push_back('k');
    }

    float ob_gain = 0.0f, ob_thr = 0.0f;
    int ob_idx = -1;
    const int n_dirs = dirs_n;
    ctx->scratch_dir_gain.assign(n_dirs, 0.0f);
    ctx->scratch_dir_thr.assign(n_dirs, 0.0f);
    auto& dir_gain = ctx->scratch_dir_gain;
    auto& dir_thr  = ctx->scratch_dir_thr;

    // Routing truncates every direction to its first D_SUB_MAX nonzeros
    // (collect_nonzero_stack); sanitize candidates the same way up front so
    // the batched dense projections match what routing will later compute.
    for (int di = 0; di < dirs_n; di++) {
      float* GF_RESTRICT wv = dirs_flat + (size_t)di * D;
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
    ctx->Pj_workspace.resize((size_t)n_dirs * ne);
    float* Pj_ptr = ctx->Pj_workspace.data();
#if GF_HAVE_BLAS
    {
      // dirs_flat already IS the flat Wd matrix layout (dirs_flat[di*D+f]).
      // Pass it directly to BLAS — eliminates the Wd allocation + memcpy loop.
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n_dirs, ne, D,
                  1.0f, dirs_flat, D, Xe_ptr, D, 0.0f, Pj_ptr, ne);
    }
#else
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n_dirs > 1)
#endif
    for (int di = 0; di < n_dirs; di++) {
      SparseVec sv;
      collect_nonzero_stack(dirs_flat + (size_t)di * D, D, sv);
      float* GF_RESTRICT prow = Pj_ptr + (size_t)di * ne;
      for (int si = 0; si < ne; si++)
        prow[si] = sparse_dot_stack(sv, Xe_ptr + (size_t)si * D);
    }
#endif

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (n_dirs > 1)
#endif
    for (int di = 0; di < n_dirs; di++) {
      SparseVec sv;
      collect_nonzero_stack(dirs_flat + (size_t)di * D, D, sv);
      if (sv.size <= 1) continue;

      const float* GF_RESTRICT proj_e = Pj_ptr + (size_t)di * ne;
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
        if (Hcs < min_child_weight || Hrs < min_child_weight) continue;
        float gain = e_base;
        for (int c = 0; c < K; c++) {
          float Gr = eGt[c] - lGc[c], Hr = eHt[c] - lHc[c];
          double lGc_l1 = thr_l1(lGc[c], reg_alpha);
          double Gr_l1 = thr_l1(Gr, reg_alpha);
          gain += 0.5f * (float)(lGc_l1 * lGc_l1 / (lHc[c] + reg_lambda + EPS) +
                                 Gr_l1 * Gr_l1 / (Hr + reg_lambda + EPS));
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

    collect_nonzero_stack(dirs_flat + (size_t)ob_idx * D, D, sv);
    std::vector<float> proj_full(ns);
    float exact_gain;
    if (exact_eval) {
      // samp_e == samp here, so the batched panel projections are exactly
      // the per-sample projections routing needs.
      std::memcpy(proj_full.data(), Pj_ptr + (size_t)ob_idx * ne,
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
      if (HLs < min_child_weight || HRs < min_child_weight) return cand_gain[t];
      exact_gain = total_base;
      for (int c = 0; c < K; c++) {
        float GR = Gt[c] - GL[c], HR = Ht[c] - HL[c];
        double gl_l1 = thr_l1(GL[c], reg_alpha);
        double gr_l1 = thr_l1(GR, reg_alpha);
        exact_gain += 0.5f * (float)(gl_l1 * gl_l1 / (HL[c] + reg_lambda + EPS) +
                                     gr_l1 * gr_l1 / (HR + reg_lambda + EPS));
      }
    }

    if (exact_gain > cand_gain[t]) {
      cand_gain[t] = exact_gain;
      cand_thr[t] = ob_thr;
      // Copy best direction from flat buffer into cand_w[t].
      cand_w[t].assign(dirs_flat + (size_t)ob_idx * D,
                       dirs_flat + (size_t)ob_idx * D + D);
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
      // Generate feat_mask once per node; reuse across both axis and oblique scans.
      std::vector<char> feat_mask(D, 1);
      if (colsample_bynode < 1.0f) {
        std::mt19937 col_rng(seed + t_node);
        feat_mask.assign(D, 0);
        int n_sel = std::max(1, (int)std::lrintf(D * colsample_bynode));
        std::vector<int> f_indices(D);
        std::iota(f_indices.begin(), f_indices.end(), 0);
        std::shuffle(f_indices.begin(), f_indices.end(), col_rng);
        for (int idx = 0; idx < n_sel; idx++) feat_mask[f_indices[idx]] = 1;
      }
      float ag = eval_axis(t_node, feat_mask);
      float g2 = eval_oblique(t_node, feat_mask);
      oblique_done[t_node] = 1;
      (void)ag;
      if (g2 < gamma || g2 <= 0.0f) {
        recycle_hist(node_hist[t_node]);
        continue;
      }

      if (!frontier.empty() && g2 < frontier.top().first) {
        frontier.push({g2, t_node});
        continue;
      }
    }

    if (cand_gain[t_node] < gamma) {
      recycle_hist(node_hist[t_node]);
      continue;
    }
    const auto& samp = node_samp[t_node];
    int tl = 2 * t_node + 1, tr_node = 2 * t_node + 2;
    int depth_t = get_node_depth(t_node);

    // --- Split Refinement Step (Removed for Optimized Joint Scan in eval_axis) ---
    if (cand_axis[t_node] < 0) ctx->cache_direction(cand_w[t_node]);
    tree->is_leaf[t_node] = 0;
    tree->split_threshold[t_node] = cand_thr[t_node];
    tree->split_gain[t_node] = cand_gain[t_node];
    tree->default_left[t_node] = cand_default_left[t_node];
    std::copy(cand_w[t_node].begin(), cand_w[t_node].end(),
              tree->split_weights.data() + (size_t)t_node * D);
    splits_left--;

    int ax = cand_axis[t_node];
    float thr = cand_thr[t_node];
    std::vector<int> left_sub, right_sub;
    std::vector<float> GL(K, 0.0f), HL(K, 0.0f);
    float PL = 0.0f;
    int ns = (int)samp.size();
    
    SparseVec sv;
    collect_nonzero_stack(cand_w[t_node].data(), D, sv);

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
          bool go_left;
          if (has_nan_in_split(ctx->is_nan.data(), j, D, sv)) {
            go_left = (tree->default_left[t_node] != 0);
          } else {
            float proj_val = sparse_dot_stack(sv, Xt + (size_t)j * D);
            go_left = (proj_val < thr);
          }
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
      std::vector<int> offset_L(actual_threads + 1, 0);
      std::vector<int> offset_R(actual_threads + 1, 0);
      double PLd = 0.0;
      for (int t = 0; t < actual_threads; t++) {
        offset_L[t + 1] = offset_L[t] + (int)tL[t].size();
        offset_R[t + 1] = offset_R[t] + (int)tR[t].size();
        for (int c = 0; c < K; c++) {
          GL[c] += tGL[t][c];
          HL[c] += tHL[t][c];
        }
        PLd += tPL[t];
      }
      left_sub.resize(offset_L[actual_threads]);
      right_sub.resize(offset_R[actual_threads]);

#pragma omp parallel num_threads(actual_threads)
      {
        int tid = omp_get_thread_num();
        if (tid < actual_threads) {
          if (!tL[tid].empty()) {
            std::memcpy(left_sub.data() + offset_L[tid], tL[tid].data(),
                        tL[tid].size() * sizeof(int));
          }
          if (!tR[tid].empty()) {
            std::memcpy(right_sub.data() + offset_R[tid], tR[tid].data(),
                        tR[tid].size() * sizeof(int));
          }
        }
      }
      PL = (float)PLd;
    } else
#endif
    {
      left_sub.reserve(ns / 2 + 64);
      right_sub.reserve(ns / 2 + 64);
      for (int si = 0; si < ns; si++) {
        int j = samp[si];
        bool go_left;
        if (has_nan_in_split(ctx->is_nan.data(), j, D, sv)) {
          go_left = (tree->default_left[t_node] != 0);
        } else {
          float proj_val = sparse_dot_stack(sv, Xt + (size_t)j * D);
          go_left = (proj_val < thr);
        }
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
    // Parent's sample index list is no longer needed after split;
    // releasing it here caps peak memory to O(N * current_depth).
    node_samp[t_node].clear();
    node_samp[t_node].shrink_to_fit();
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
        float raw = -thr_l1(Gs, reg_alpha) / (Hs + reg_lambda + EPS);
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
        int t = 0;
        for (int dep = 0; dep < tree->max_depth; dep++) {
          if (tree->is_leaf[t]) break;
          bool has_nan = false;
          const SparseVec& sv = node_nz[t];
          if (has_nan_in_split(ctx->is_nan.data(), i, D, sv)) {
            has_nan = true;
          }
          if (has_nan) {
            t = (tree->default_left[t] != 0) ? (2 * t + 1) : (2 * t + 2);
          } else {
            float proj = sparse_dot_stack(sv, Xt + (size_t)i * D);
            t = (proj < tree->split_threshold[t]) ? (2 * t + 1) : (2 * t + 2);
          }
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
  // Fast path: purely numerical data (D_cat == 0).
  // The Xt copy loop only trivially copies numeric columns as-is;
  // _gf_route handles NaN via tree->na_means directly, so skip the allocation.
  if (D_num == D) {
    _gf_route(tree, X, N, out_pred);
    return;
  }
  // General path: categorical columns need rank-encoding into Xt.
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
        ti[f] = v; // Keep NaN as NaN!
      } else {
        int fc = f - D_num;
        if (std::isnan(v) || fc >= (int)tree->cat_ranks.size()) {
          ti[f] = std::nanf(""); // Represent missing category as NaN!
        } else {
          const auto& m = tree->cat_ranks[fc];
          auto it = m.find((int)std::lrintf(v));
          ti[f] = (it != m.end()) ? it->second : std::nanf(""); // Represent unseen category as NaN!
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
  uint8_t default_left = 1;
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
      ti[f] = xi[f]; // Keep NaN as NaN!
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

  std::vector<int> col_offset(D_cat + 1, 0);
  int r_sum = 0;
  for (int fc = 0; fc < D_cat; fc++) {
    col_offset[fc] = r_sum;
    r_sum += cat_card[fc] + 1;
  }
  col_offset[D_cat] = r_sum;

  static constexpr int TILE = 16;
  std::vector<std::vector<GFCompactNode>> tile_nodes(TILE);
  std::vector<std::vector<float>> tile_leaves(TILE);
  // Flattened rank table: size TILE * r_sum
  std::vector<float> tile_rank_flat(D_cat > 0 ? (size_t)TILE * r_sum : 0, 0.0f);
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
          nodes[c].default_left = tr->default_left[h];
          next += 2;
        }
      }

      // ── per-tree rank tables (code → encoded value) ─────────────────────
      if (D_cat > 0) {
        int base_offset = tt * r_sum;
        for (int fc = 0; fc < D_cat; fc++) {
          int offset = base_offset + col_offset[fc];
          float fallback = std::nanf(""); // Represent missing/unseen category as NaN!
          std::fill(tile_rank_flat.begin() + offset, tile_rank_flat.begin() + offset + cat_card[fc] + 1, fallback);
          if (fc < (int)tr->cat_ranks.size()) {
            for (const auto& kv : tr->cat_ranks[fc]) {
              auto it = raw2code[fc].find(kv.first);
              if (it != raw2code[fc].end()) {
                tile_rank_flat[offset + it->second] = kv.second;
              }
            }
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
            const float* GF_RESTRICT tr_rank = tile_rank_flat.data() + (size_t)tt * r_sum;
            for (int fc = 0; fc < D_cat; fc++)
              row[D_num + fc] = tr_rank[col_offset[fc] + ci[fc]];
          }
          const GFCompactNode* GF_RESTRICT nd = tile_nodes[tt].data();
          int n = 0;
          while (nd[n].left >= 0) {
            bool has_nan = false;
            const SparseVec& sv = nd[n].nz;
            if (sv.size == 1) {
              int idx = sv.indices[0];
              if (idx < D_num && std::isnan(rp[idx])) {
                has_nan = true;
              }
            }
            if (has_nan) {
              n = (nd[n].default_left != 0) ? nd[n].left : nd[n].right;
            } else {
              float proj = 0.0f;
              if (sv.size == 1) {
                proj = sv.values[0] * rp[sv.indices[0]];
              } else {
                const float* na_means = trees[tile_lo + tt]->na_means.data();
                for (int k = 0; k < sv.size; k++) {
                  float v = rp[sv.indices[k]];
                  if (std::isnan(v)) {
                    v = na_means[sv.indices[k]];
                  }
                  proj += sv.values[k] * v;
                }
              }
              n = (proj < nd[n].thr) ? nd[n].left : nd[n].right;
            }
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
                          uint8_t* is_leaf, uint8_t* default_left) {
  const OQTree* tree = static_cast<const OQTree*>(handle);
  int n = tree->total_nodes, K = tree->K;
  for (int i = 0; i < n; ++i) split_hyp_idx[i] = tree->split_hyp_idx[i];
  for (int i = 0; i < n; ++i) split_threshold[i] = tree->split_threshold[i];
  for (size_t i = 0; i < (size_t)n * K; ++i) leaf_values[i] = tree->leaf_values[i];
  for (int i = 0; i < n; ++i) is_leaf[i] = tree->is_leaf[i];
  if (default_left) {
    if (!tree->default_left.empty()) {
      for (int i = 0; i < n; ++i) default_left[i] = tree->default_left[i];
    } else {
      for (int i = 0; i < n; ++i) default_left[i] = 1;
    }
  }
}

GF_API void* oqtree_from_arrays(const int* split_hyp_idx,
                                const float* split_threshold,
                                const float* leaf_values,
                                const uint8_t* is_leaf,
                                const uint8_t* default_left, int total_nodes, int K,
                                int max_depth) {
  OQTree* tree = new OQTree();
  tree->total_nodes = total_nodes;
  tree->K = K;
  tree->max_depth = max_depth;
  tree->split_hyp_idx.assign(split_hyp_idx, split_hyp_idx + total_nodes);
  tree->split_threshold.assign(split_threshold, split_threshold + total_nodes);
  tree->leaf_values.assign(leaf_values, leaf_values + (size_t)total_nodes * K);
  tree->is_leaf.assign(is_leaf, is_leaf + total_nodes);
  if (default_left) {
    tree->default_left.assign(default_left, default_left + total_nodes);
  } else {
    tree->default_left.assign(total_nodes, 1);
  }
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

    // K=2 fast path: p0 + p1 = 1, so only one exp() is needed.
    if (K == 2) {
      float diff = F[offset] - F[offset + 1];
      float e = std::exp(-std::abs(diff));
      float p0 = (diff >= 0.0f) ? (1.0f / (1.0f + e)) : (e / (1.0f + e));
      float p1 = 1.0f - p0;
      G[offset]     = p0 - oh[offset];
      H[offset]     = p0 * p1;
      G[offset + 1] = p1 - oh[offset + 1];
      H[offset + 1] = p0 * p1;  // symmetric
      continue;
    }

    // General path: find max for numerical stability
    float fmax = F[offset];
    for (int c = 1; c < K; c++) {
      if (F[offset + c] > fmax) fmax = F[offset + c];
    }

    // Cache exp results in p_buf to avoid computing exp() twice per sample.
    // Stack-allocated: zero heap traffic in the hot loop.
    float p_buf[64];  // K <= 64 in all practical boosting scenarios
    double sum_exp = 0.0;
    const int Kc = (K < 64) ? K : 64;
    for (int c = 0; c < Kc; c++) {
      p_buf[c] = std::exp(F[offset + c] - fmax);
      sum_exp += p_buf[c];
    }
    float inv_sum = 1.0f / (float)sum_exp;
    for (int c = 0; c < Kc; c++) {
      float p = p_buf[c] * inv_sum;
      G[offset + c] = p - oh[offset + c];
      H[offset + c] = p * (1.0f - p);
    }
  }
}

}  // extern "C"