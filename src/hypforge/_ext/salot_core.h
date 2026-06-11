// salot_core.h — internal shared core for salot.cpp / evopool.cpp
// Constants, sparse helpers, and the standardized CD direction solver.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32) || defined(_WIN64)
#define SALOT_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define SALOT_API __attribute__((visibility("default")))
#else
#define SALOT_API
#endif

#if defined(_MSC_VER)
#define SALOT_RESTRICT __restrict
#else
#define SALOT_RESTRICT __restrict__
#endif

static constexpr float EPS = 1e-8f;
static constexpr float MIN_CHILD_W = 0.1f;
static constexpr int D_SUB_MAX = 32;     // feature dims entering one CD solve
static constexpr int HIST_BINS = 64;     // local bins for candidate scans
static constexpr int AX_BINS = 256;      // global pre-binned feature codes
static constexpr int EST_NE_MAX = 4096;  // gain-estimation subsample cap
static constexpr int CD_SWEEPS = 4;      // max CD sweeps (warm-started)
static constexpr int WLS_CAP = 1024;     // CD-WLS instance subsample cap
static constexpr int SCREEN_N = 1024;    // SIS screening subsample

static inline void collect_nonzero(const float* w, int n,
                                   std::vector<std::pair<int, float>>& out) {
  out.clear();
  for (int f = 0; f < n; f++)
    if (w[f] != 0.0f) out.emplace_back(f, w[f]);
}

static inline float sparse_dot(const std::vector<std::pair<int, float>>& nz,
                               const float* SALOT_RESTRICT xi) {
  float proj = 0.0f;
  for (const auto& p : nz) proj += p.second * xi[p.first];
  return proj;
}

// dot of two sparse vectors (index-ascending pairs)
static inline float sparse_sparse_dot(
    const std::vector<std::pair<int, float>>& a,
    const std::vector<std::pair<int, float>>& b) {
  float s = 0.0f;
  size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i].first < b[j].first)
      i++;
    else if (a[i].first > b[j].first)
      j++;
    else {
      s += a[i].second * b[j].second;
      i++;
      j++;
    }
  }
  return s;
}

// ─── Standardized streaming-CD direction solver → nested sparsity path ──────
//
// Solves  argmin_w  Σ_i [ h_i (wᵀx_i)²/2 + g_i (wᵀx_i) ] + λ‖w‖²/2
// restricted to feat_sub by warm-started Gauss-Seidel coordinate descent
// over a contiguous n×b panel — O(sweeps·n·b), no Gram matrix, no Cholesky.
// λ may be 0 (the diagonal Σh·x² keeps updates well-defined).
//
// In-panel STANDARDIZATION (real-data critical): column j is scaled to unit
// hessian-weighted second moment, so λ acts scale-free and the whitened
// energy reduces to w̃². Weights are mapped back before output.
//
// Output: up to 4 unit-norm directions along the nested whitened-energy path
// {0.995, 0.95, 0.90, 0.80} (a lasso-path analogue) — candidates spanning
// the dof-vs-fit trade-off from ONE solve. 1-sparse levels are skipped
// (raw features cover them).
static void node_cd_candidates(const std::vector<int>& samp,
                               const std::vector<int>& feat_sub, int D,
                               const float* SALOT_RESTRICT X,
                               const float* SALOT_RESTRICT G,
                               const float* SALOT_RESTRICT H, int K,
                               int k_class, float reg_lambda,
                               std::vector<std::vector<float>>& out_dirs) {
  out_dirs.clear();
  int n = (int)samp.size(), b = (int)feat_sub.size();
  if (n < 2 * b) return;

  std::vector<float> P((size_t)n * b);
  std::vector<float> g(n), h(n);
  for (int i = 0; i < n; i++) {
    int idx = samp[i];
    const float* SALOT_RESTRICT xi = X + (size_t)idx * D;
    float* SALOT_RESTRICT pi = P.data() + (size_t)i * b;
    for (int j = 0; j < b; j++) pi[j] = xi[feat_sub[j]];
    g[i] = G[(size_t)idx * K + k_class];
    h[i] = H[(size_t)idx * K + k_class];
  }

  std::vector<float> Add(b, 0.0f), cg(b, 0.0f);
  float Wh = 0.0f;
  for (int i = 0; i < n; i++) {
    const float* SALOT_RESTRICT pi = P.data() + (size_t)i * b;
    float hi = h[i], gi = g[i];
    Wh += hi;
    for (int j = 0; j < b; j++) {
      Add[j] += hi * pi[j] * pi[j];
      cg[j] += pi[j] * gi;
    }
  }
  if (Wh < 1e-12f) return;

  std::vector<float> sclj(b, 1.0f);
  for (int j = 0; j < b; j++) {
    if (Add[j] < 1e-20f) continue;
    sclj[j] = std::sqrt(Add[j] / Wh);
    cg[j] /= sclj[j];
    Add[j] = Wh;
  }
  for (int i = 0; i < n; i++) {
    float* SALOT_RESTRICT pi = P.data() + (size_t)i * b;
    for (int j = 0; j < b; j++) pi[j] /= sclj[j];
  }

  // Warm start at the univariate ridge solutions; CD resolves only the
  // cross-correlations, so early stopping triggers in a few sweeps.
  std::vector<float> w(b, 0.0f), f(n, 0.0f);
  for (int j = 0; j < b; j++)
    if (Add[j] > 1e-12f) w[j] = -cg[j] / (Add[j] + reg_lambda);
  for (int i = 0; i < n; i++) {
    const float* SALOT_RESTRICT pi = P.data() + (size_t)i * b;
    float fi = 0.0f;
    for (int j = 0; j < b; j++) fi += pi[j] * w[j];
    f[i] = fi;
  }
  for (int s = 0; s < CD_SWEEPS; s++) {
    float max_rel = 0.0f;
    for (int j = 0; j < b; j++) {
      if (Add[j] < 1e-12f) continue;
      float sj = 0.0f;
      for (int i = 0; i < n; i++) sj += h[i] * P[(size_t)i * b + j] * f[i];
      float wnew = (-cg[j] - sj + w[j] * Add[j]) / (Add[j] + reg_lambda);
      float dw = wnew - w[j];
      if (dw != 0.0f) {
        for (int i = 0; i < n; i++) f[i] += dw * P[(size_t)i * b + j];
        w[j] = wnew;
        float rel = std::abs(dw) / (std::abs(wnew) + 1e-12f);
        if (rel > max_rel) max_rel = rel;
      }
    }
    if (max_rel < 1e-3f) break;
  }

  // Standardized coords → whitened energy is just w̃².
  std::vector<int> ord(b);
  std::iota(ord.begin(), ord.end(), 0);
  std::vector<float> ev(b);
  float total_ev = 0.0f;
  for (int j = 0; j < b; j++) {
    ev[j] = w[j] * w[j];
    total_ev += ev[j];
  }
  if (total_ev < 1e-20f) return;
  std::sort(ord.begin(), ord.end(),
            [&](int a, int c) { return ev[a] > ev[c]; });

  static const float LEVELS[2] = {0.95f, 0.80f};
  int cur_keep = b;
  for (float level : LEVELS) {
    float cum = 0.0f;
    int keep = b;
    for (int r = 0; r < b; r++) {
      cum += ev[ord[r]];
      if (cum >= level * total_ev) {
        keep = r + 1;
        break;
      }
    }
    if (keep >= cur_keep && cur_keep != b) continue;
    for (int r = keep; r < cur_keep; r++) {
      int j = ord[r];
      if (w[j] != 0.0f) {
        float dw = -w[j];
        for (int i = 0; i < n; i++) f[i] += dw * P[(size_t)i * b + j];
        w[j] = 0.0f;
      }
    }
    cur_keep = keep;
    for (int r = 0; r < keep; r++) {  // one refinement sweep on actives
      int j = ord[r];
      if (Add[j] < 1e-12f) continue;
      float sj = 0.0f;
      for (int i = 0; i < n; i++) sj += h[i] * P[(size_t)i * b + j] * f[i];
      float wnew = (-cg[j] - sj + w[j] * Add[j]) / (Add[j] + reg_lambda);
      float dw = wnew - w[j];
      if (dw != 0.0f) {
        for (int i = 0; i < n; i++) f[i] += dw * P[(size_t)i * b + j];
        w[j] = wnew;
      }
    }
    if (keep < 2) break;

    std::vector<float> w_orig(b, 0.0f);
    float norm = 0.0f;
    for (int j = 0; j < b; j++) {
      if (w[j] != 0.0f) w_orig[j] = w[j] / sclj[j];
      norm += w_orig[j] * w_orig[j];
    }
    norm = std::sqrt(norm);
    if (norm < 1e-12f) continue;
    std::vector<float> w_full(D, 0.0f);
    for (int j = 0; j < b; j++)
      if (w_orig[j] != 0.0f) w_full[feat_sub[j]] = w_orig[j] / norm;
    out_dirs.push_back(std::move(w_full));
  }
}

// SIS screening scores on a subsample: s_d = |Σ x_d g| / sqrt(Σ h x_d² + λ).
static inline void sis_scores(const std::vector<int>& samp, int n_use,
                              const float* SALOT_RESTRICT X, int D, int D_num,
                              const float* SALOT_RESTRICT G,
                              const float* SALOT_RESTRICT H, int K, int k_class,
                              float reg_lambda, std::vector<float>& fscore) {
  fscore.assign(D_num, 0.0f);
  std::vector<float> cg_s(D_num, 0.0f), add_s(D_num, 0.0f);
  int n_scr = std::min((int)samp.size(), n_use);
  for (int i = 0; i < n_scr; i++) {
    int idx = samp[i];
    const float* SALOT_RESTRICT xi = X + (size_t)idx * D;
    float gi = G[(size_t)idx * K + k_class];
    float hi = H[(size_t)idx * K + k_class];
    for (int d = 0; d < D_num; d++) {
      cg_s[d] += xi[d] * gi;
      add_s[d] += hi * xi[d] * xi[d];
    }
  }
  for (int d = 0; d < D_num; d++)
    fscore[d] = std::abs(cg_s[d]) / std::sqrt(add_s[d] + reg_lambda + EPS);
}

// Class with the largest gradient mass on the subsample.
static inline int dominant_class(const std::vector<int>& samp,
                                 const float* SALOT_RESTRICT G, int K) {
  int best_c = 0;
  float best_mass = -1.0f;
  for (int c = 0; c < K; c++) {
    float m = 0.0f;
    for (int i : samp) m += std::abs(G[(size_t)i * K + c]);
    if (m > best_mass) {
      best_mass = m;
      best_c = c;
    }
  }
  return best_c;
}