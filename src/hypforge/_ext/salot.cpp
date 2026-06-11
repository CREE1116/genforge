// salot.cpp — SALOT v9: context-cached, subtraction-based oblique booster
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
//   * Binning context computed ONCE in salot_ctx_create, reused per round
//     (numeric codes static; only cat columns are re-coded per round).
//   * Histogram subtraction (smaller child fresh, larger = parent − smaller).
//   * Lazy A* best-first growth with admissible node potential.
//   * Oblique directions only where ns ≥ OBLIQUE_MIN.

#include <queue>

#include "bfstree_types.h"
#include "salot_core.h"

static constexpr int OBLIQUE_MIN = 64;  // min node size to fit directions

// ─── Binning context (per dataset, reused across all boosting rounds) ───────
struct SalotCtx {
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

  // Persistent direction cache (the pool-persistence trait shared by the
  // two high-balanced-accuracy models, HypForge and EvoPool): oblique
  // directions that won a split stay available as FREE candidates (no CD
  // solve) at every node of every subsequent round. NUMERIC-ONLY: cat
  // rank coordinates change meaning each round. Ring buffer,
  // cosine-deduplicated, deterministic.
  static constexpr int DIR_CACHE_MAX = 32;
  std::vector<std::vector<float>> dir_cache;  // dense D each
  int dir_cache_next = 0;

  void cache_direction(const std::vector<float>& w) {
    for (int d = D_num; d < D; d++)
      if (w[d] != 0.0f) return;  // touches a cat dim — meaning is per-round
    float nw = 0.0f;
    for (int d = 0; d < D; d++) nw += w[d] * w[d];
    if (nw < 1e-12f) return;
    for (const auto& c : dir_cache) {
      float dot = 0.0f;
      for (int d = 0; d < D; d++) dot += c[d] * w[d];
      if (std::abs(dot) / std::sqrt(nw) > 0.95f) return;  // redundant
    }
    std::vector<float> wn(w.begin(), w.begin() + D);
    if ((int)dir_cache.size() < DIR_CACHE_MAX) {
      dir_cache.push_back(std::move(wn));
    } else {
      dir_cache[dir_cache_next] = std::move(wn);
      dir_cache_next = (dir_cache_next + 1) % DIR_CACHE_MAX;
    }
  }
};

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
static void _salot_route(const BFSTree* tree, const float* X, int N,
                         float* out_pred) {
  int D = tree->D, K = tree->K;
  int T = tree->total_nodes;
  std::vector<std::vector<std::pair<int, float>>> node_nz(T);
  for (int t = 0; t < T; t++) {
    if (tree->is_leaf[t]) continue;
    collect_nonzero(tree->split_weights.data() + (size_t)t * D, D, node_nz[t]);
  }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++) {
    const float* SALOT_RESTRICT xi = X + (size_t)i * D;
    int t = 0;
    for (int dep = 0; dep < tree->max_depth; dep++) {
      if (tree->is_leaf[t]) break;
      float proj = sparse_dot(node_nz[t], xi);
      t = (proj < tree->split_threshold[t]) ? (2 * t + 1) : (2 * t + 2);
    }
    const float* lv = tree->leaf_values.data() + (size_t)t * K;
    float* oi = out_pred + (size_t)i * K;
    for (int k = 0; k < K; k++) oi[k] = lv[k];
  }
}

extern "C" {

// Pre-bin all features once. Numeric: NaN → μ_f imputation, then linear
// 256-bin grid. Categorical (columns [D_num, D)): build the raw-value
// dictionary; rank codes are written per round by salot_build_v8.
// X is fully copied into the context (it need not stay alive).
SALOT_API void* salot_ctx_create(const float* X, int N, int D, int D_num,
                                 const int* sub, int Ns) {
  auto* ctx = new SalotCtx();
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
    const float* SALOT_RESTRICT xi = X + (size_t)sub[si] * D;
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
    const float* SALOT_RESTRICT xi = X + (size_t)i * D;
    float* SALOT_RESTRICT ti = ctx->Ximp.data() + (size_t)i * D;
    uint8_t* SALOT_RESTRICT ci = ctx->code.data() + (size_t)i * D;
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

  // ── categorical: value dictionary (sorted → deterministic dense ids) ──────
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
      int nan_id = (int)vals.size();  // NaN is its own (always last) category
      ctx->cat_card[fc] = nan_id + 1;
      for (int i = 0; i < N; i++) {
        float v = X[(size_t)i * D + f];
        ctx->cat_dense[(size_t)i * ctx->D_cat + fc] =
            std::isnan(v) ? nan_id : m[(int)std::lrintf(v)];
      }
    }
  }
  return static_cast<void*>(ctx);
}

SALOT_API void salot_ctx_free(void* h) { delete static_cast<SalotCtx*>(h); }

// ─── salot_build_v8 — one boosting round on a binning context ────────────────
// (name kept for ABI continuity; implements the v9 semantics above)
SALOT_API void* salot_build_v8(void* ctx_handle, const float* X, const float* G,
                               const float* H, int K, const int* sub, int Ns,
                               int max_depth, float reg_lambda,
                               float* out_pred) {
  (void)X;  // v9 reads only the context's transformed copy
  auto* ctx = static_cast<SalotCtx*>(ctx_handle);
  const int D = ctx->D, D_num = ctx->D_num, D_cat = ctx->D_cat, N = ctx->N;
  const float* SALOT_RESTRICT Xt = ctx->Ximp.data();
  const int STRIDE = 2 * K + 1;
  const size_t HSZ = (size_t)D * AX_BINS * STRIDE;

  // Best-first capacity: leaf budget 2^max_depth, internal depth up to 2x.
  const int internal_depth = std::min(2 * max_depth, 22);
  const int max_leaves = 1 << max_depth;
  int max_nodes = (1 << (internal_depth + 1)) - 1;
  auto* tree = new BFSTree();
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
  std::copy(ctx->col_mean.begin(), ctx->col_mean.end(),
            tree->na_means.begin());

  // ── per-round categorical re-encoding: signed Newton-score ranks ──────────
  // score(v) = Σ_{i∈v} g_{i,k*} / (Σ_{i∈v} h_{i,k*} + λ), k* = dominant class.
  // Monotone rank substitution makes each cat column an ordinary ordinal
  // feature for this round: same binning, same axis scan, same oblique w·x̃.
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
      if (card <= 1) {  // no observed values at all
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
      float* SALOT_RESTRICT Xw = ctx->Ximp.data();
      uint8_t* SALOT_RESTRICT cw = ctx->code.data();
      const int32_t* SALOT_RESTRICT cd = ctx->cat_dense.data();
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
      // NaN-category rank doubles as the unseen-category fallback.
      tree->na_means[f] = rank_of[card - 1];
    }
  }
  const uint8_t* SALOT_RESTRICT code = ctx->code.data();

  // Parallel histogram accumulation: thread-local buffers, then a serial
  // merge in tid order. Static scheduling fixes which samples each tid
  // sums, and the tid-ordered merge fixes the float addition order, so the
  // result is DETERMINISTIC across runs (for a given thread count).
  auto accumulate_hist = [&](const int* rows, int nr, float* SALOT_RESTRICT hb,
                             float* node_P_out) {
    double P_acc = 0.0;
#ifdef _OPENMP
    if (nr >= 4096) {
      int nth = omp_get_max_threads();
      std::vector<std::vector<float>> tbuf(nth);
      std::vector<double> tP(nth, 0.0);
#pragma omp parallel num_threads(nth)
      {
        int tid = omp_get_thread_num();
        tbuf[tid].assign(HSZ, 0.0f);
        float* SALOT_RESTRICT lb = tbuf[tid].data();
        double lp = 0.0;
#pragma omp for schedule(static) nowait
        for (int si = 0; si < nr; si++) {
          int j = rows[si];
          const uint8_t* SALOT_RESTRICT cj = code + (size_t)j * D;
          const float* SALOT_RESTRICT gj = G + (size_t)j * K;
          const float* SALOT_RESTRICT hj = H + (size_t)j * K;
          for (int f = 0; f < D; f++) {
            float* SALOT_RESTRICT slot =
                lb + ((size_t)f * AX_BINS + cj[f]) * STRIDE;
            for (int c = 0; c < K; c++) {
              slot[c] += gj[c];
              slot[K + c] += hj[c];
            }
            slot[2 * K] += 1.0f;
          }
          if (node_P_out)
            for (int c = 0; c < K; c++)
              lp += 0.5 * (double)gj[c] * gj[c] /
                    ((double)hj[c] + reg_lambda + EPS);
        }
        tP[tid] = lp;
      }
      // Single-region merge: each slot sums its per-thread partials in tid
      // order — deterministic and one parallel region total.
#pragma omp parallel for schedule(static)
      for (int64_t i = 0; i < (int64_t)HSZ; i++) {
        float s = hb[i];
        for (int t = 0; t < nth; t++) s += tbuf[t][i];
        hb[i] = s;
      }
      for (int t = 0; t < nth; t++) P_acc += tP[t];
      if (node_P_out) *node_P_out = (float)P_acc;
      return;
    }
#endif
    for (int si = 0; si < nr; si++) {
      int j = rows[si];
      const uint8_t* SALOT_RESTRICT cj = code + (size_t)j * D;
      const float* SALOT_RESTRICT gj = G + (size_t)j * K;
      const float* SALOT_RESTRICT hj = H + (size_t)j * K;
      for (int f = 0; f < D; f++) {
        float* SALOT_RESTRICT slot =
            hb + ((size_t)f * AX_BINS + cj[f]) * STRIDE;
        for (int c = 0; c < K; c++) {
          slot[c] += gj[c];
          slot[K + c] += hj[c];
        }
        slot[2 * K] += 1.0f;
      }
      if (node_P_out)
        for (int c = 0; c < K; c++)
          P_acc +=
              0.5 * (double)gj[c] * gj[c] / ((double)hj[c] + reg_lambda + EPS);
    }
    if (node_P_out) *node_P_out = (float)P_acc;
  };

  std::vector<std::vector<int>> node_samp(max_nodes);
  std::vector<std::vector<float>> node_hist(max_nodes);
  std::vector<float> node_G((size_t)max_nodes * K, 0.0f);
  std::vector<float> node_H((size_t)max_nodes * K, 0.0f);
  std::vector<float> node_P(max_nodes,
                            0.0f);  // Σ_i Σ_c g²/(h+λ): gain UPPER BOUND
  std::vector<char> node_has_tot(max_nodes, 0);

  // Pending split candidates (best-first frontier).
  std::vector<float> cand_gain(max_nodes, 0.0f);
  std::vector<float> cand_thr(max_nodes, 0.0f);
  std::vector<int> cand_axis(max_nodes, -1);
  std::vector<int> cand_bcode(max_nodes, 0);
  std::vector<std::vector<float>> cand_w(max_nodes);
  std::vector<std::vector<float>> cand_proj(
      max_nodes);  // oblique routing cache

  std::vector<char> oblique_done(max_nodes, 0);

  // ── Phase 1 (cheap, at node creation): exact axis tournament from the
  // histogram. Its gain is a LOWER BOUND of the node's true best gain and
  // serves as the frontier priority.
  auto eval_axis = [&](int t) -> float {
    int ns = (int)node_samp[t].size();
    const float* SALOT_RESTRICT hb = node_hist[t].data();

    std::vector<float> Gt(K, 0.0f), Ht(K, 0.0f);
    for (int b = 0; b < AX_BINS; b++) {
      const float* SALOT_RESTRICT slot = hb + (size_t)b * STRIDE;
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

    // Axis tournament: exact, from the (possibly subtracted) histogram.
    // Parallel over features; merge is deterministic regardless of thread
    // count (max gain, ties broken by the smallest feature index — the same
    // winner the serial scan picks).
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
        const float* SALOT_RESTRICT fbuf = hb + (size_t)f * AX_BINS * STRIDE;
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

  // ── Phase 2 (expensive, only when the node is actually popped to split):
  // oblique tournament. Candidates are evaluated EXACTLY on the full node
  // sample (a subsampled estimate rescaled by mass inflates gain variance
  // and wins by noise). Upgrades the stored candidate when better.
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
    std::vector<float> Gc(K), Hc(K);

    std::vector<int> wls_samp = stride_cap(samp, WLS_CAP);

    // Class targets for the CD solves. The Newton WLS objective is
    // per-class; with K ≥ 3 the top-2 gradient-mass classes each get their
    // own screening + CD solve (one-vs-rest), so directions serving the
    // runner-up class structure are not crowded out by the dominant class.
    // K = 2 is symmetric (G_0 = −G_1): one solve covers both.
    std::vector<std::pair<float, int>> cmass(K);
    for (int c = 0; c < K; c++) cmass[c] = {0.0f, c};
    for (int i : wls_samp)
      for (int c = 0; c < K; c++)
        cmass[c].first += std::abs(G[(size_t)i * K + c]);
    std::sort(cmass.begin(), cmass.end(),
              [](const std::pair<float, int>& a,
                 const std::pair<float, int>& b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
              });
    const int n_cls = (K >= 3) ? 2 : 1;
    // Screening and CD span ALL D dims: cat columns participate via their
    // per-round rank coordinates (the CD panel standardizes each column, so
    // the arbitrary 0..card-1 rank scale is harmless).
    std::vector<std::vector<float>> dirs;
    std::vector<float> fscore;
    for (int cc = 0; cc < n_cls; cc++) {
      int kc = cmass[cc].second;
      sis_scores(wls_samp, SCREEN_N, Xt, D, D, G, H, K, kc, reg_lambda,
                 fscore);
      int b_sel = std::min(D_SUB_MAX, D);
      std::vector<int> feat_sub(D);
      std::iota(feat_sub.begin(), feat_sub.end(), 0);
      if (b_sel < D) {
        std::partial_sort(feat_sub.begin(), feat_sub.begin() + b_sel,
                          feat_sub.end(),
                          [&](int a, int c) { return fscore[a] > fscore[c]; });
        feat_sub.resize(b_sel);
      }
      std::vector<std::vector<float>> cdirs;
      node_cd_candidates(wls_samp, feat_sub, D, Xt, G, H, K, kc, reg_lambda,
                         cdirs);
      for (auto& w : cdirs) dirs.push_back(std::move(w));
    }
    // Persistent cached directions compete as free candidates. Selection is
    // GAIN-based, not proxy-based: a dominant-class SIS-overlap filter
    // systematically discards exactly the minority-region directions the
    // cache exists to preserve. Pre-stage: every cached direction gets a
    // very cheap gain estimate (512-stride sample, 16 bins, all classes);
    // the top-2 join stage-1 as free candidates. Deterministic.
    {
      int nc = (int)ctx->dir_cache.size();
      if (nc > 0) {
        std::vector<int> samp_p = stride_cap(samp, 512);
        int np = (int)samp_p.size();
        std::vector<float> pGt(K, 0.0f), pHt(K, 0.0f);
        for (int j : samp_p)
          for (int c = 0; c < K; c++) {
            pGt[c] += G[(size_t)j * K + c];
            pHt[c] += H[(size_t)j * K + c];
          }
        float p_base = 0.0f;
        for (int c = 0; c < K; c++)
          p_base -= 0.5f * pGt[c] * pGt[c] / (pHt[c] + reg_lambda + EPS);

        constexpr int PRE_BINS = 16;
        std::vector<float> pproj(np);
        std::vector<float> pG((size_t)PRE_BINS * K), pH((size_t)PRE_BINS * K);
        std::vector<int> pc(PRE_BINS);
        std::vector<std::pair<int, float>> nz_p;
        std::vector<std::pair<float, int>> rank;
        rank.reserve(nc);
        for (int ci = 0; ci < nc; ci++) {
          collect_nonzero(ctx->dir_cache[ci].data(), D, nz_p);
          if (nz_p.size() <= 1) continue;
          float mn = 1e30f, mx = -1e30f;
          for (int si = 0; si < np; si++) {
            float v = sparse_dot(nz_p, Xt + (size_t)samp_p[si] * D);
            pproj[si] = v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
          }
          if (mx - mn < 1e-12f) continue;
          std::fill(pG.begin(), pG.end(), 0.0f);
          std::fill(pH.begin(), pH.end(), 0.0f);
          std::fill(pc.begin(), pc.end(), 0);
          float scale = (float)PRE_BINS / (mx - mn + EPS);
          for (int si = 0; si < np; si++) {
            int j = samp_p[si];
            int b = (int)((pproj[si] - mn) * scale);
            if (b < 0) b = 0;
            if (b >= PRE_BINS) b = PRE_BINS - 1;
            pc[b]++;
            size_t bo = (size_t)b * K;
            for (int c = 0; c < K; c++) {
              pG[bo + c] += G[(size_t)j * K + c];
              pH[bo + c] += H[(size_t)j * K + c];
            }
          }
          std::fill(Gc.begin(), Gc.end(), 0.0f);
          std::fill(Hc.begin(), Hc.end(), 0.0f);
          int n_left = 0;
          float best = 0.0f;
          for (int b = 0; b < PRE_BINS - 1; b++) {
            n_left += pc[b];
            size_t bo = (size_t)b * K;
            for (int c = 0; c < K; c++) {
              Gc[c] += pG[bo + c];
              Hc[c] += pH[bo + c];
            }
            if (n_left < 5 || np - n_left < 5) continue;
            float gain = p_base;
            for (int c = 0; c < K; c++) {
              float Gr = pGt[c] - Gc[c], Hr = pHt[c] - Hc[c];
              gain += 0.5f * (Gc[c] * Gc[c] / (Hc[c] + reg_lambda + EPS) +
                              Gr * Gr / (Hr + reg_lambda + EPS));
            }
            if (gain > best) best = gain;
          }
          if (best > 0.0f) rank.push_back({best, ci});
        }
        int n_take = std::min((int)rank.size(), 2);
        std::partial_sort(
            rank.begin(), rank.begin() + n_take, rank.end(),
            [](const std::pair<float, int>& a, const std::pair<float, int>& b) {
              if (a.first != b.first) return a.first > b.first;
              return a.second < b.second;
            });
        for (int r = 0; r < n_take; r++) {
          const auto& cw = ctx->dir_cache[rank[r].second];
          std::vector<float> wfull(D, 0.0f);
          std::copy(cw.begin(), cw.end(), wfull.begin());
          dirs.push_back(std::move(wfull));
        }
      }
    }
    if (dirs.empty()) return cand_gain[t];

    // Stage 1 — select the best oblique (candidate, threshold) on a capped
    // subsample: selection AMONG oblique candidates shares one noise level,
    // so it is internally fair and O(1) in ns.
    std::vector<int> samp_e = stride_cap(samp, EST_NE_MAX);
    int ne = (int)samp_e.size();
    bool exact_eval = (ne == ns);
    std::vector<float> eGt(K, 0.0f), eHt(K, 0.0f);
    if (exact_eval) {
      eGt = Gt;
      eHt = Ht;
    } else {
      for (int j : samp_e)
        for (int c = 0; c < K; c++) {
          eGt[c] += G[(size_t)j * K + c];
          eHt[c] += H[(size_t)j * K + c];
        }
    }
    float e_base = 0.0f;
    for (int c = 0; c < K; c++)
      e_base -= 0.5f * eGt[c] * eGt[c] / (eHt[c] + reg_lambda + EPS);

    float ob_gain = 0.0f, ob_thr = 0.0f;
    int ob_idx = -1;
    std::vector<float> ob_proj_e;
    const int n_dirs = (int)dirs.size();
    // Candidates are independent: evaluate them in parallel and merge by
    // (gain, lowest index) — the same winner the serial scan picks.
    std::vector<float> dir_gain(n_dirs, 0.0f), dir_thr(n_dirs, 0.0f);
    std::vector<std::vector<float>> dir_proj(n_dirs);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (n_dirs > 1)
#endif
    for (int di = 0; di < n_dirs; di++) {
      std::vector<std::pair<int, float>> nz_c;
      collect_nonzero(dirs[di].data(), D, nz_c);
      if (nz_c.size() <= 1) continue;  // axis scan covers 1-sparse

      std::vector<float> proj_e(ne);
      float min_v = 1e30f, max_v = -1e30f;
      for (int si = 0; si < ne; si++) {
        float proj = sparse_dot(nz_c, Xt + (size_t)samp_e[si] * D);
        proj_e[si] = proj;
        if (proj < min_v) min_v = proj;
        if (proj > max_v) max_v = proj;
      }
      if (max_v - min_v < 1e-12f) continue;

      std::vector<float> bin_G((size_t)HIST_BINS * K, 0.0f);
      std::vector<float> bin_H((size_t)HIST_BINS * K, 0.0f);
      std::vector<int> bin_cnt(HIST_BINS, 0);
      float scale = (float)HIST_BINS / (max_v - min_v + EPS);
      for (int si = 0; si < ne; si++) {
        int j = samp_e[si];
        int b = (int)((proj_e[si] - min_v) * scale);
        if (b < 0) b = 0;
        if (b >= HIST_BINS) b = HIST_BINS - 1;
        bin_cnt[b]++;
        size_t bo = (size_t)b * K;
        for (int c = 0; c < K; c++) {
          bin_G[bo + c] += G[(size_t)j * K + c];
          bin_H[bo + c] += H[(size_t)j * K + c];
        }
      }

      std::vector<float> lGc(K, 0.0f), lHc(K, 0.0f);
      int n_left = 0;
      float d_gain = 0.0f, d_thr = 0.0f;
      for (int b = 0; b < HIST_BINS - 1; b++) {
        n_left += bin_cnt[b];
        size_t bo = (size_t)b * K;
        for (int c = 0; c < K; c++) {
          lGc[c] += bin_G[bo + c];
          lHc[c] += bin_H[bo + c];
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
        dir_proj[di] = std::move(proj_e);
      }
    }
    for (int di = 0; di < n_dirs; di++) {
      if (dir_gain[di] > ob_gain) {
        ob_gain = dir_gain[di];
        ob_thr = dir_thr[di];
        ob_idx = di;
      }
    }
    if (ob_idx < 0) return cand_gain[t];
    ob_proj_e = std::move(dir_proj[ob_idx]);
    std::vector<std::pair<int, float>> nz_c;

    // Stage 2 — EXACT re-evaluation of the single winner on the full node
    // sample (one projection pass; it doubles as the routing projection, so
    // its marginal cost over routing is ~zero). The axis-vs-oblique decision
    // is exact-vs-exact: no subsample-noise wins.
    collect_nonzero(dirs[ob_idx].data(), D, nz_c);
    std::vector<float> proj_full;
    float exact_gain;
    if (exact_eval) {
      proj_full = std::move(ob_proj_e);
      exact_gain = ob_gain;
    } else {
      proj_full.resize(ns);
      std::vector<float> GL(K, 0.0f), HL(K, 0.0f);
      int n_left = 0;
#ifdef _OPENMP
      if (ns >= 8192) {
        // tid-ordered merge keeps the float sums deterministic.
        int nth = omp_get_max_threads();
        std::vector<std::vector<float>> tGL(nth), tHL(nth);
        std::vector<int> tnl(nth, 0);
#pragma omp parallel num_threads(nth)
        {
          int tid = omp_get_thread_num();
          tGL[tid].assign(K, 0.0f);
          tHL[tid].assign(K, 0.0f);
          float* SALOT_RESTRICT lGL = tGL[tid].data();
          float* SALOT_RESTRICT lHL = tHL[tid].data();
          int lnl = 0;
#pragma omp for schedule(static) nowait
          for (int si = 0; si < ns; si++) {
            int j = samp[si];
            float proj = sparse_dot(nz_c, Xt + (size_t)j * D);
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
        for (int t2 = 0; t2 < nth; t2++) {
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
          float proj = sparse_dot(nz_c, Xt + (size_t)j * D);
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
      cand_proj[t] = std::move(proj_full);  // reuse for routing (samp order)
    }
    return cand_gain[t];
  };

  // ── Best-first growth: split the highest-gain leaf first. ─────────────────
  // Leaf budget = 2^user_depth (same capacity as a full level-wise tree),
  // allocated where the gains are; internal depth cap = 2×user_depth.
  node_samp[0].assign(sub, sub + Ns);
  {
    node_hist[0].assign(HSZ, 0.0f);
    accumulate_hist(sub, Ns, node_hist[0].data(), &node_P[0]);
  }

  std::priority_queue<std::pair<float, int>> frontier;
  if (Ns >= 20) frontier.push({node_P[0], 0});

  int splits_left = max_leaves - 1;
  while (splits_left > 0 && !frontier.empty()) {
    int t_node = frontier.top().second;
    frontier.pop();

    // Lazily materialize this node's histogram: build the SMALLER of
    // {node, sibling} fresh, derive the larger from the parent buffer,
    // then free the parent. Histogram work is thus paid only along the
    // lineage of nodes that actually split.
    if (node_hist[t_node].empty()) {
      int par = (t_node - 1) / 2;
      int sib = (t_node % 2 == 1) ? t_node + 1 : t_node - 1;
      bool self_small = node_samp[t_node].size() <= node_samp[sib].size();
      int t_small = self_small ? t_node : sib;
      int t_large = self_small ? sib : t_node;

      node_hist[t_small].assign(HSZ, 0.0f);
      float* SALOT_RESTRICT hs = node_hist[t_small].data();
      accumulate_hist(node_samp[t_small].data(),
                      (int)node_samp[t_small].size(), hs, nullptr);
      float* SALOT_RESTRICT hp = node_hist[par].data();
      for (size_t i = 0; i < HSZ; i++) hp[i] -= hs[i];
      node_hist[t_large] = std::move(node_hist[par]);
    }

    // Lazy A*: node_P is an ADMISSIBLE upper bound on the node's split
    // gain, so popping by bound, evaluating exactly once, and re-queueing
    // with the exact gain when it no longer tops the frontier reproduces
    // EXACT best-first order — with at most one evaluation per node
    // (never more work than eager evaluation, usually much less).
    if (!oblique_done[t_node]) {
      float ag = eval_axis(t_node);
      float g2 = eval_oblique(t_node);
      oblique_done[t_node] = 1;
      (void)ag;
      if (g2 <= 0.0f) {
        node_hist[t_node].clear();
        node_hist[t_node].shrink_to_fit();
        continue;
      }
      if (!frontier.empty() && g2 < frontier.top().first) {
        frontier.push({g2, t_node});
        continue;
      }
    }
    const auto& samp = node_samp[t_node];
    int tl = 2 * t_node + 1, tr_node = 2 * t_node + 2;
    int depth_t = 0;
    while (((1 << (depth_t + 1)) - 1) <= t_node) depth_t++;

    // Apply the stored split; persist winning oblique directions.
    if (cand_axis[t_node] < 0) ctx->cache_direction(cand_w[t_node]);
    tree->is_leaf[t_node] = 0;
    tree->split_threshold[t_node] = cand_thr[t_node];
    tree->split_gain[t_node] = cand_gain[t_node];
    std::copy(cand_w[t_node].begin(), cand_w[t_node].end(),
              tree->split_weights.data() + (size_t)t_node * D);
    splits_left--;

    // Partition: codes for axis splits, cached projections for oblique.
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
      // Parallel partition. Static scheduling assigns each thread one
      // contiguous index range in tid order, so concatenating the
      // per-thread outputs in tid order reproduces the serial sample order
      // exactly.
      int nth = omp_get_max_threads();
      std::vector<std::vector<int>> tL(nth), tR(nth);
      std::vector<std::vector<float>> tGL(nth), tHL(nth);
      std::vector<double> tPL(nth, 0.0);
#pragma omp parallel num_threads(nth)
      {
        int tid = omp_get_thread_num();
        auto& Lv = tL[tid];
        auto& Rv = tR[tid];
        tGL[tid].assign(K, 0.0f);
        tHL[tid].assign(K, 0.0f);
        float* SALOT_RESTRICT gl = tGL[tid].data();
        float* SALOT_RESTRICT hl = tHL[tid].data();
        double pl = 0.0;
#pragma omp for schedule(static) nowait
        for (int si = 0; si < ns; si++) {
          int j = samp[si];
          bool go_left = (ax >= 0)
                             ? (code[(size_t)j * D + ax] <= (uint8_t)bcode)
                             : (proj[si] < thr);
          if (go_left) {
            Lv.push_back(j);
            const float* SALOT_RESTRICT gj = G + (size_t)j * K;
            const float* SALOT_RESTRICT hj = H + (size_t)j * K;
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
      for (int t = 0; t < nth; t++) {
        nl += tL[t].size();
        nr += tR[t].size();
      }
      left_sub.reserve(nl);
      right_sub.reserve(nr);
      double PLd = 0.0;
      for (int t = 0; t < nth; t++) {
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
      for (int si = 0; si < ns; si++) {
        int j = samp[si];
        bool go_left = (ax >= 0) ? (code[(size_t)j * D + ax] <= (uint8_t)bcode)
                                 : (proj[si] < thr);
        if (go_left) {
          left_sub.push_back(j);
          const float* SALOT_RESTRICT gj = G + (size_t)j * K;
          const float* SALOT_RESTRICT hj = H + (size_t)j * K;
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

    // Children enter the frontier with a FREE priority — their loss
    // potential 0.5·Σ G_c²/(H_c+λ) computed from the totals accumulated
    // during partitioning. No histogram is built here: histograms are
    // built lazily only when a node is actually popped to split, so the
    // ~half of children that end as leaves cost nothing.
    node_samp[tl] = std::move(left_sub);
    node_samp[tr_node] = std::move(right_sub);
    bool can_deepen = (depth_t + 1 < internal_depth) && (splits_left > 0);
    if (can_deepen) {
      for (int child : {tl, tr_node}) {
        int cns = (int)node_samp[child].size();
        if (cns < 20) continue;
        // Priority = Σ_i Σ_c g²/(h+λ): the gain of giving every sample its
        // own leaf — a true UPPER BOUND on any split sequence below the
        // node (A*-style optimistic best-first). Crucially it does NOT
        // vanish when gradients cancel (ΣG≈0 across an oblique boundary),
        // which is exactly where the best splits live.
        if (node_P[child] > 0.0f) frontier.push({node_P[child], child});
      }
    } else {
      node_hist[t_node].clear();
      node_hist[t_node].shrink_to_fit();
    }
  }
  for (int t = 0; t < max_nodes; t++) {
    node_hist[t].clear();
    node_hist[t].shrink_to_fit();
  }

  // ── Leaves from stored totals: path smoothing scaled by lambda ────────────
  {
    std::vector<float> sm((size_t)max_nodes * K, 0.0f);
    std::vector<char> hasv(max_nodes, 0);
    for (int t = 0; t < max_nodes; t++) {
      if (!node_has_tot[t] && node_samp[t].empty()) continue;
      if (!node_has_tot[t]) {  // node skipped before totals were derived
        for (int j : node_samp[t])
          for (int c = 0; c < K; c++) {
            node_G[(size_t)t * K + c] += G[(size_t)j * K + c];
            node_H[(size_t)t * K + c] += H[(size_t)j * K + c];
          }
      }
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

  if (out_pred) {
    std::memset(out_pred, 0, (size_t)N * K * sizeof(float));
    // Build-side rows already know their leaf (node_samp partition) — only
    // rows outside the training subsample need routing.
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
      std::vector<std::vector<std::pair<int, float>>> node_nz(max_nodes);
      for (int t = 0; t < max_nodes; t++) {
        if (tree->is_leaf[t]) continue;
        collect_nonzero(tree->split_weights.data() + (size_t)t * D, D,
                        node_nz[t]);
      }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (int i = 0; i < N; i++) {
        if (in_sub[i]) continue;
        const float* SALOT_RESTRICT xi = Xt + (size_t)i * D;
        int t = 0;
        for (int dep = 0; dep < tree->max_depth; dep++) {
          if (tree->is_leaf[t]) break;
          float proj = sparse_dot(node_nz[t], xi);
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

// One-shot compatibility wrapper (signature unchanged; seed is unused — v9
// is deterministic by construction).
SALOT_API void* salot_build_v7(const float* X, int N, int D, int D_num,
                               const float* G, const float* H, int K,
                               const int* sub, int Ns, int max_depth,
                               float reg_lambda, unsigned int seed,
                               float* out_pred) {
  (void)seed;
  void* ctx = salot_ctx_create(X, N, D, D_num, sub, Ns);
  void* tree =
      salot_build_v8(ctx, X, G, H, K, sub, Ns, max_depth, reg_lambda, out_pred);
  salot_ctx_free(ctx);
  return tree;
}

// Predict on RAW X: NaN and categorical values are re-encoded into the
// tree's transformed space (numeric NaN → stored μ_f; cat value → stored
// per-tree rank; NaN/unseen cat → stored NaN-category rank), then routed.
SALOT_API void salot_predict(void* tree_handle, const float* X, int N, int K,
                             float* out_pred) {
  const BFSTree* tree = static_cast<const BFSTree*>(tree_handle);
  std::memset(out_pred, 0, (size_t)N * K * sizeof(float));
  if (!tree) return;
  const int D = tree->D;
  if ((int)tree->na_means.size() != D) {  // legacy tree: route raw
    _salot_route(tree, X, N, out_pred);
    return;
  }
  const int D_num = tree->D_num;
  std::vector<float> Xt((size_t)N * D);
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int i = 0; i < N; i++) {
    const float* SALOT_RESTRICT xi = X + (size_t)i * D;
    float* SALOT_RESTRICT ti = Xt.data() + (size_t)i * D;
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
  _salot_route(tree, Xt.data(), N, out_pred);
}

SALOT_API void salot_tree_free(void* tree_handle) {
  delete static_cast<BFSTree*>(tree_handle);
}

// ─── tree meta (de)serialization: μ_f, cat rank maps, D_num ─────────────────
// sizes[0] = D_num, sizes[1] = D_cat, sizes[2] = total cat-map entries,
// sizes[3] = len(na_means) (0 for legacy trees).
SALOT_API void salot_tree_meta_sizes(void* tree_handle, int* sizes) {
  const BFSTree* tree = static_cast<const BFSTree*>(tree_handle);
  sizes[0] = tree->D_num;
  sizes[1] = (int)tree->cat_ranks.size();
  int total = 0;
  for (const auto& m : tree->cat_ranks) total += (int)m.size();
  sizes[2] = total;
  sizes[3] = (int)tree->na_means.size();
}

SALOT_API void salot_tree_export_meta(void* tree_handle, float* na_means,
                                      int* cat_sizes, int* cat_keys,
                                      float* cat_vals) {
  const BFSTree* tree = static_cast<const BFSTree*>(tree_handle);
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

SALOT_API void salot_tree_import_meta(void* tree_handle, int D_num,
                                      const float* na_means, int na_len,
                                      const int* cat_sizes, int D_cat,
                                      const int* cat_keys,
                                      const float* cat_vals) {
  BFSTree* tree = static_cast<BFSTree*>(tree_handle);
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

}  // extern "C"
