// gos.cpp — GOS (d-GOS) tree interface
// Separated from salot.cpp. Compiled into the same shared library.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "bfstree_types.h"

static constexpr float EPS_H = 1e-8f;

// Forward declaration — defined in bfstree.cpp, same shared library.
BFSTree* bfs_build_best_first(const float* const* hyp_caches, int P,
                              const std::vector<int>& train_indices,
                              const float* G_full, const float* H_full, int K,
                              int max_depth, int min_split, int min_leaf,
                              float reg_lambda);

static void _gos_apply_tree(const BFSTree* tree, const float* Z, int N, int K,
                            float* out_pred) {
  std::vector<int> node_assign(N, 0);
  for (int depth = 0; depth < tree->max_depth; depth++) {
    int n_nodes = 1 << depth, base = n_nodes - 1;
    for (int local = 0; local < n_nodes; local++) {
      int t = base + local;
      if (tree->is_leaf[t]) continue;
      int b = tree->split_hyp_idx[t];
      float thr = tree->split_threshold[t];
      const float* zb = Z + (size_t)b * N;
      int tl = 2 * t + 1, tr = 2 * t + 2;
      for (int i = 0; i < N; i++)
        if (node_assign[i] == t) node_assign[i] = (zb[i] < thr) ? tl : tr;
    }
  }
  for (int i = 0; i < N; i++) {
    const float* lv = tree->leaf_values.data() + (size_t)node_assign[i] * K;
    float* oi = out_pred + (size_t)i * K;
    for (int k = 0; k < K; k++) oi[k] = lv[k];
  }
}

extern "C" {

void* gos_build_tree(const float* Z, int N, int B, int K, const float* G_full,
                     const float* H_full, const int* tree_sub, int Ns_tree,
                     int max_depth, float reg_lambda, float* out_pred) {
  static constexpr int TOP_M = 128;
  int B_use = std::min(B, TOP_M);
  std::vector<const float*> caches(B_use);
  for (int b = 0; b < B_use; b++) caches[b] = Z + (size_t)b * N;
  std::vector<int> tree_indices(tree_sub, tree_sub + Ns_tree);
  BFSTree* tree =
      bfs_build_best_first(caches.data(), B_use, tree_indices, G_full, H_full,
                           K, max_depth, 20, 10, reg_lambda);
  if (!tree) return nullptr;
  _gos_apply_tree(tree, Z, N, K, out_pred);
  return static_cast<void*>(tree);
}

void gos_predict_tree(void* tree_handle, const float* Z, int N, int B, int K,
                      float* out_pred) {
  const BFSTree* tree = static_cast<const BFSTree*>(tree_handle);
  if (!tree) {
    std::memset(out_pred, 0, (size_t)N * K * sizeof(float));
    return;
  }
  _gos_apply_tree(tree, Z, N, K, out_pred);
}

void bfs_tree_free(void* tree_handle) {
  delete static_cast<BFSTree*>(tree_handle);
}

void gos_evolve(const float* X, int Ns, int D, const float* G_full,
                const float* H_full, int K, const float* centers, int M,
                float* W, int B, float tau, float eta, float lam,
                float gamma_rep, int n_steps) {
  std::vector<float> grads(B * D, 0.0f);
  std::vector<float> repel_grads(B * D, 0.0f);

  for (int step = 0; step < n_steps; ++step) {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int b = 0; b < B; ++b) {
      const float* wb = W + (size_t)b * D;
      float* gb = grads.data() + (size_t)b * D;

      std::vector<float> z(Ns, 0.0f);
      for (int n = 0; n < Ns; ++n) {
        const float* xn = X + (size_t)n * D;
        float s = 0.0f;
        for (int d = 0; d < D; ++d) s += xn[d] * wb[d];
        z[n] = s;
      }

      std::vector<float> A(Ns * M), mu_c(Ns, 0.0f);
      for (int n = 0; n < Ns; ++n) {
        float* An = A.data() + n * M;
        float max_d = -1e30f;
        for (int m = 0; m < M; ++m) {
          float diff = z[n] - centers[m];
          float val = -(diff * diff) / tau;
          An[m] = val;
          if (val > max_d) max_d = val;
        }
        float sum_e = 0.0f;
        for (int m = 0; m < M; ++m) {
          An[m] = expf(An[m] - max_d);
          sum_e += An[m];
        }
        float inv = 1.0f / (sum_e + 1e-10f);
        float mu = 0.0f;
        for (int m = 0; m < M; ++m) {
          An[m] *= inv;
          mu += An[m] * centers[m];
        }
        mu_c[n] = mu;
      }

      std::vector<float> g_hat(M * K, 0.0f), h_hat(M * K, 0.0f);
      for (int n = 0; n < Ns; ++n) {
        const float* An = A.data() + n * M;
        const float* Gn = G_full + (size_t)n * K;
        const float* Hn = H_full + (size_t)n * K;
        for (int m = 0; m < M; ++m) {
          float a = An[m];
          for (int k = 0; k < K; ++k) {
            g_hat[m * K + k] += a * Gn[k];
            h_hat[m * K + k] += a * Hn[k];
          }
        }
      }

      std::vector<float> G_cum(M * K), H_cum(M * K);
      for (int k = 0; k < K; ++k) G_cum[k] = g_hat[k], H_cum[k] = h_hat[k];
      for (int m = 1; m < M; ++m)
        for (int k = 0; k < K; ++k) {
          G_cum[m * K + k] = G_cum[(m - 1) * K + k] + g_hat[m * K + k];
          H_cum[m * K + k] = H_cum[(m - 1) * K + k] + h_hat[m * K + k];
        }

      std::vector<float> f(M * K);
      for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
          float g = G_cum[m * K + k], h = H_cum[m * K + k];
          f[m * K + k] = fabsf(g) / sqrtf(fabsf(h) + EPS_H);
        }

      std::vector<float> alpha(M * K);
      for (int k = 0; k < K; ++k) {
        float max_f = -1e30f;
        for (int m = 0; m < M; ++m)
          if (f[m * K + k] > max_f) max_f = f[m * K + k];
        float sum_e = 0.0f;
        for (int m = 0; m < M; ++m) {
          alpha[m * K + k] = expf(f[m * K + k] - max_f);
          sum_e += alpha[m * K + k];
        }
        float inv = 1.0f / (sum_e + 1e-10f);
        for (int m = 0; m < M; ++m) alpha[m * K + k] *= inv;
      }

      std::vector<float> phi(M * K, 0.0f), psi(M * K, 0.0f);
      for (int m = M - 1; m >= 0; --m) {
        for (int k = 0; k < K; ++k) {
          float g = G_cum[m * K + k], h = H_cum[m * K + k];
          float sqh = sqrtf(fabsf(h) + EPS_H), h15 = sqh * sqh * sqh;
          float al = alpha[m * K + k];
          phi[m * K + k] = al * (g >= 0 ? 1.0f : -1.0f) / (sqh * K) +
                           (m < M - 1 ? phi[(m + 1) * K + k] : 0.0f);
          psi[m * K + k] = al * (-0.5f * fabsf(g)) / (h15 * K) +
                           (m < M - 1 ? psi[(m + 1) * K + k] : 0.0f);
        }
      }

      std::vector<float> grad_z(Ns, 0.0f);
      for (int n = 0; n < Ns; ++n) {
        const float* An = A.data() + n * M;
        const float* Gn = G_full + (size_t)n * K;
        const float* Hn = H_full + (size_t)n * K;
        float gz = 0.0f;
        for (int m = 0; m < M; ++m) {
          float c_minus_mu = centers[m] - mu_c[n];
          float dot = 0.0f;
          for (int k = 0; k < K; ++k)
            dot += Gn[k] * phi[m * K + k] + Hn[k] * psi[m * K + k];
          gz += An[m] * c_minus_mu * dot;
        }
        grad_z[n] = (2.0f / tau) * gz;
      }

      for (int d = 0; d < D; ++d) gb[d] = 0.0f;
      for (int n = 0; n < Ns; ++n) {
        const float* xn = X + (size_t)n * D;
        float gz = grad_z[n];
        for (int d = 0; d < D; ++d) gb[d] += xn[d] * gz;
      }
    }

    std::vector<float> norms(B), Wn(B * D);
    for (int b = 0; b < B; ++b) {
      const float* wb = W + (size_t)b * D;
      float s = 0.0f;
      for (int d = 0; d < D; ++d) s += wb[d] * wb[d];
      norms[b] = sqrtf(s) + 1e-9f;
      for (int d = 0; d < D; ++d) Wn[b * D + d] = wb[d] / norms[b];
    }
    std::vector<float> C(B * B, 0.0f);
    for (int i = 0; i < B; ++i)
      for (int j = 0; j < B; ++j) {
        if (i == j) continue;
        float s = 0.0f;
        for (int d = 0; d < D; ++d) s += Wn[i * D + d] * Wn[j * D + d];
        C[i * B + j] = s;
      }
    for (int b = 0; b < B; ++b) {
      float* rg = repel_grads.data() + (size_t)b * D;
      for (int d = 0; d < D; ++d) rg[d] = 0.0f;
      for (int b2 = 0; b2 < B; ++b2) {
        if (b2 == b) continue;
        float cos_val = C[b * B + b2], cos2 = cos_val * cos_val;
        float denom = 1.0f - cos2 + 1e-6f;
        float coeff = 2.0f * cos_val / denom;
        for (int d = 0; d < D; ++d)
          rg[d] += coeff * (Wn[b2 * D + d] - cos_val * Wn[b * D + d]) / norms[b];
      }
    }
    for (int b = 0; b < B; ++b) {
      float* wb = W + (size_t)b * D;
      float* gb = grads.data() + (size_t)b * D;
      float* rg = repel_grads.data() + (size_t)b * D;
      float thr = lam * eta;
      for (int d = 0; d < D; ++d) wb[d] += eta * (gb[d] - gamma_rep * rg[d]);
      for (int d = 0; d < D; ++d) {
        float v = wb[d];
        wb[d] = (v > thr ? v - thr : v < -thr ? v + thr : 0.0f);
      }
      float norm = 0.0f;
      for (int d = 0; d < D; ++d) norm += wb[d] * wb[d];
      norm = sqrtf(norm);
      if (norm > 1e-5f) {
        float inv = 1.0f / norm;
        for (int d = 0; d < D; ++d) wb[d] *= inv;
      } else {
        for (int d = 0; d < D; ++d) wb[d] = Wn[b * D + d];
      }
    }
  }
}

}  // extern "C"
