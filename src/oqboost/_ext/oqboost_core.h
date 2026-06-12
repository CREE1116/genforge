// oqboost_core.h — internal shared core for oqboost.cpp
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
  int indices[D_SUB_MAX];
  float values[D_SUB_MAX];
};

static inline void collect_nonzero_stack(const float* w, int n, SparseVec& sv) {
  sv.size = 0;
  for (int f = 0; f < n; f++) {
    if (w[f] != 0.0f) {
      sv.indices[sv.size] = f;
      sv.values[sv.size] = w[f];
      sv.size++;
      if (sv.size >= D_SUB_MAX) break;
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