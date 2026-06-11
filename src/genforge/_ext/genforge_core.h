// genforge_core.h — internal shared core for genforge.cpp
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
#define GF_API __declspec(dllexport)
#elif defined(__GNUC__) || defined(__clang__)
#define GF_API __attribute__((visibility("default")))
#else
#define GF_API
#endif

#if defined(_MSC_VER)
#define GF_RESTRICT __restrict
#else
#define GF_RESTRICT __restrict__
#endif

static constexpr float EPS = 1e-8f;
static constexpr float MIN_CHILD_W = 0.1f;
static constexpr int D_SUB_MAX = 16;     // feature dims entering one CD solve
static constexpr int HIST_BINS = 64;     // local bins for candidate scans
static constexpr int AX_BINS = 256;      // global pre-binned feature codes
static constexpr int EST_NE_MAX = 2048;  // gain-estimation subsample cap
static constexpr int CD_SWEEPS = 2;      // max CD sweeps (warm-started)
static constexpr int WLS_CAP = 2048;     // CD-WLS instance subsample cap
static constexpr int SCREEN_N = 2048;    // SIS screening subsample

static inline void collect_nonzero(const float* w, int n,
                                   std::vector<std::pair<int, float>>& out) {
  out.clear();
  for (int f = 0; f < n; f++)
    if (w[f] != 0.0f) out.emplace_back(f, w[f]);
}

static inline float sparse_dot(const std::vector<std::pair<int, float>>& nz,
                               const float* GF_RESTRICT xi) {
  float proj = 0.0f;
  for (const auto& p : nz) proj += p.second * xi[p.first];
  return proj;
}

struct SparseVec {
  int size = 0;
  int indices[16];
  float values[16];
};

static inline void collect_nonzero_stack(const float* w, int n, SparseVec& sv) {
  sv.size = 0;
  for (int f = 0; f < n; f++) {
    if (w[f] != 0.0f) {
      sv.indices[sv.size] = f;
      sv.values[sv.size] = w[f];
      sv.size++;
      if (sv.size >= 16) break;
    }
  }
}

static inline float sparse_dot_stack(const SparseVec& sv,
                                     const float* GF_RESTRICT xi) {
  switch (sv.size) {
    case 1:
      return sv.values[0] * xi[sv.indices[0]];
    case 2:
      return sv.values[0] * xi[sv.indices[0]] +
             sv.values[1] * xi[sv.indices[1]];
    case 3:
      return sv.values[0] * xi[sv.indices[0]] +
             sv.values[1] * xi[sv.indices[1]] +
             sv.values[2] * xi[sv.indices[2]];
    case 4:
      return sv.values[0] * xi[sv.indices[0]] +
             sv.values[1] * xi[sv.indices[1]] +
             sv.values[2] * xi[sv.indices[2]] +
             sv.values[3] * xi[sv.indices[3]];
    case 5:
      return sv.values[0] * xi[sv.indices[0]] +
             sv.values[1] * xi[sv.indices[1]] +
             sv.values[2] * xi[sv.indices[2]] +
             sv.values[3] * xi[sv.indices[3]] +
             sv.values[4] * xi[sv.indices[4]];
    case 6:
      return sv.values[0] * xi[sv.indices[0]] +
             sv.values[1] * xi[sv.indices[1]] +
             sv.values[2] * xi[sv.indices[2]] +
             sv.values[3] * xi[sv.indices[3]] +
             sv.values[4] * xi[sv.indices[4]] +
             sv.values[5] * xi[sv.indices[5]];
    default:
      float proj = 0.0f;
      for (int i = 0; i < sv.size; i++) {
        proj += sv.values[i] * xi[sv.indices[i]];
      }
      return proj;
  }
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
                               const std::vector<float>& wls_w,
                               const std::vector<int>& feat_sub, int D,
                               const float* GF_RESTRICT X,
                               const float* GF_RESTRICT G,
                               const float* GF_RESTRICT H, int K,
                               int k_class, float reg_lambda,
                               std::vector<std::vector<float>>& out_dirs) {
  out_dirs.clear();
  int n = (int)samp.size(), b = (int)feat_sub.size();
  if (n < 2 * b) return;

  // Transpose: P is column-major [b × n] for cache locality and auto-vectorization
  std::vector<float> P((size_t)b * n);
  std::vector<float> g(n), h(n);
  for (int i = 0; i < n; i++) {
    int idx = samp[i];
    const float* GF_RESTRICT xi = X + (size_t)idx * D;
    for (int j = 0; j < b; j++) {
      P[(size_t)j * n + i] = xi[feat_sub[j]];
    }
    g[i] = G[(size_t)idx * K + k_class] * wls_w[i];
    h[i] = H[(size_t)idx * K + k_class] * wls_w[i];
  }

  std::vector<float> Add(b, 0.0f), cg(b, 0.0f);
  float Wh = 0.0f;
  for (int j = 0; j < b; j++) {
    const float* GF_RESTRICT pj = P.data() + (size_t)j * n;
    double sum_add = 0.0, sum_cg = 0.0;
    for (int i = 0; i < n; i++) {
      float p_val = pj[i];
      sum_add += (double)h[i] * p_val * p_val;
      sum_cg += (double)p_val * g[i];
    }
    Add[j] = (float)sum_add;
    cg[j] = (float)sum_cg;
  }
  for (int i = 0; i < n; i++) {
    Wh += h[i];
  }
  if (Wh < 1e-12f) return;

  std::vector<float> sclj(b, 1.0f);
  for (int j = 0; j < b; j++) {
    if (Add[j] < 1e-20f) continue;
    sclj[j] = std::sqrt(Add[j] / Wh);
    cg[j] /= sclj[j];
    Add[j] = Wh;
  }
  for (int j = 0; j < b; j++) {
    float* GF_RESTRICT pj = P.data() + (size_t)j * n;
    float inv_scl = 1.0f / (sclj[j] + EPS);
    for (int i = 0; i < n; i++) {
      pj[i] *= inv_scl;
    }
  }

  // 1. Calculate the Gram matrix A using Column-Major contiguous memory
  std::vector<float> A((size_t)b * b, 0.0f);
  for (int j = 0; j < b; j++) {
    A[(size_t)j * b + j] = Wh;
  }
  for (int j = 0; j < b; j++) {
    if (Add[j] < 1e-12f) continue;
    const float* GF_RESTRICT pj = P.data() + (size_t)j * n;
    for (int k = j + 1; k < b; k++) {
      if (Add[k] < 1e-12f) continue;
      const float* GF_RESTRICT pk = P.data() + (size_t)k * n;
      double sum_jk = 0.0;
      for (int i = 0; i < n; i++) {
        sum_jk += (double)h[i] * pj[i] * pk[i];
      }
      float val = (float)(sum_jk / (sclj[j] * sclj[k] + EPS));
      A[(size_t)j * b + k] = val;
      A[(size_t)k * b + j] = val; // symmetric
    }
  }

  // Warm start at the univariate ridge solutions
  std::vector<float> w(b, 0.0f);
  for (int j = 0; j < b; j++) {
    if (Add[j] > 1e-12f) {
      w[j] = -cg[j] / (Wh + reg_lambda);
    }
  }

  // Coordinate Descent on the Gram matrix
  for (int s = 0; s < CD_SWEEPS; s++) {
    float max_rel = 0.0f;
    for (int j = 0; j < b; j++) {
      if (Add[j] < 1e-12f) continue;
      double sum_aw = 0.0;
      for (int k = 0; k < b; k++) {
        if (k != j) {
          sum_aw += (double)A[(size_t)j * b + k] * w[k];
        }
      }
      float wnew = (float)((-cg[j] - sum_aw) / (Wh + reg_lambda));
      float dw = wnew - w[j];
      if (dw != 0.0f) {
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
      w[j] = 0.0f;
    }
    cur_keep = keep;
    for (int r = 0; r < keep; r++) {  // one refinement sweep on actives
      int j = ord[r];
      if (Add[j] < 1e-12f) continue;
      double sum_aw = 0.0;
      for (int k = 0; k < b; k++) {
        if (k != j) {
          sum_aw += (double)A[(size_t)j * b + k] * w[k];
        }
      }
      float wnew = (float)((-cg[j] - sum_aw) / (Wh + reg_lambda));
      w[j] = wnew;
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
                              const float* GF_RESTRICT X, int D, int D_num,
                              const float* GF_RESTRICT G,
                              const float* GF_RESTRICT H, int K, int k_class,
                              float reg_lambda, std::vector<float>& fscore) {
  fscore.assign(D_num, 0.0f);
  std::vector<float> cg_s(D_num, 0.0f), add_s(D_num, 0.0f);
  int n_scr = std::min((int)samp.size(), n_use);
  for (int i = 0; i < n_scr; i++) {
    int idx = samp[i];
    const float* GF_RESTRICT xi = X + (size_t)idx * D;
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
                                 const float* GF_RESTRICT G, int K) {
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