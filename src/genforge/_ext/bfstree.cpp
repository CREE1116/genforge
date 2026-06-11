// bfstree.cpp — BFS Oblique Tree Engine (tree structures only)
//
// Layout conventions (all row-major):
//   Z          : [P, N]           Z[p*N + i]
//   thresholds : [9, P]           thresholds[q*P + p]
//   G, H       : [N, K]           G[i*K + k]
//   leaf_values: [total_nodes, K] leaf_values[t*K + k]

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <queue>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "bfstree_types.h"

// ── Histogram-based split search
// ──────────────────────────────────────────────
static const int N_HIST_BINS = 255;

// ── Legacy histogram BFS tree builder (used by the standalone bfstree_build
// API)
static BFSTree* bfs_build_hist(const float* Z_sub,  // [P, Ns]  row-major
                               const float* G, const float* H, int P, int Ns,
                               int K, int max_depth, int min_split,
                               int min_leaf, float reg_lambda) {
  BFSTree* tree = new BFSTree();
  tree->max_depth = max_depth;
  tree->K = K;
  int total_nodes = (1 << (max_depth + 1)) - 1;
  tree->total_nodes = total_nodes;
  tree->split_hyp_idx.assign(total_nodes, -1);
  tree->split_threshold.assign(total_nodes, 0.0f);
  tree->split_gain.assign(total_nodes, 0.0f);
  tree->leaf_values.assign((size_t)total_nodes * K, 0.0f);
  tree->is_leaf.assign(total_nodes, 1);

  std::vector<std::vector<int>> node_samples(total_nodes);
  node_samples[0].resize(Ns);
  std::iota(node_samples[0].begin(), node_samples[0].end(), 0);

  std::vector<float> G_sum(K), H_sum(K), G_tot(K), H_tot(K), G_L(K), H_L(K);

  for (int depth = 0; depth <= max_depth; depth++) {
    int n_nodes = 1 << depth;
    int base = n_nodes - 1;
    bool is_last = (depth == max_depth);

    for (int local = 0; local < n_nodes; local++) {
      int t = base + local;
      auto& samples = node_samples[t];
      int n_in = (int)samples.size();

      if (n_in == 0) {
        tree->is_leaf[t] = 1;
        continue;
      }

      std::fill(G_sum.begin(), G_sum.end(), 0.0f);
      std::fill(H_sum.begin(), H_sum.end(), 0.0f);
      for (int idx : samples) {
        const float* gi = G + (size_t)idx * K;
        const float* hi = H + (size_t)idx * K;
        for (int k = 0; k < K; k++) {
          G_sum[k] += gi[k];
          H_sum[k] += hi[k];
        }
      }
      float* lv = &tree->leaf_values[(size_t)t * K];
      for (int k = 0; k < K; k++) lv[k] = -(G_sum[k] / (H_sum[k] + reg_lambda));

      if (is_last || n_in < min_split) {
        tree->is_leaf[t] = 1;
        {
          std::vector<int> tmp;
          tmp.swap(samples);
        }
        continue;
      }

      const int* sp = samples.data();
      int Ns_node = n_in;
      std::vector<int> sub_buf;
      if (n_in > 25000) {
        Ns_node = 25000;
        sub_buf.assign(samples.begin(), samples.begin() + Ns_node);
        sp = sub_buf.data();
      }

      std::fill(G_tot.begin(), G_tot.end(), 0.0f);
      std::fill(H_tot.begin(), H_tot.end(), 0.0f);
      for (int j = 0; j < Ns_node; j++) {
        const float* gi = G + (size_t)sp[j] * K;
        const float* hi = H + (size_t)sp[j] * K;
        for (int k = 0; k < K; k++) {
          G_tot[k] += gi[k];
          H_tot[k] += hi[k];
        }
      }
      float parent_score = 0.0f;
      for (int k = 0; k < K; k++)
        parent_score += G_tot[k] * G_tot[k] / (H_tot[k] + reg_lambda);

      float best_gain = 1e-5f;
      int best_p = -1;
      float best_thr = 0.0f;

#ifdef _OPENMP
#pragma omp parallel
      {
        std::vector<float> G_hist(N_HIST_BINS * K);
        std::vector<float> H_hist(N_HIST_BINS * K);
        std::vector<int> cnt_hist(N_HIST_BINS);
        std::vector<float> G_Lp(K), H_Lp(K);
        float tl_gain = 1e-5f;
        int tl_p = -1;
        float tl_thr = 0.0f;
#pragma omp for nowait
        for (int p = 0; p < P; p++) {
          const float* Z_p = Z_sub + (size_t)p * Ns;
          float lmin = Z_p[sp[0]], lmax = Z_p[sp[0]];
          for (int j = 1; j < Ns_node; j++) {
            float v = Z_p[sp[j]];
            if (v < lmin) lmin = v;
            if (v > lmax) lmax = v;
          }
          float range = lmax - lmin;
          if (range < 1e-8f) continue;
          float inv_r = N_HIST_BINS / range;
          std::fill(G_hist.begin(), G_hist.end(), 0.0f);
          std::fill(H_hist.begin(), H_hist.end(), 0.0f);
          std::fill(cnt_hist.begin(), cnt_hist.end(), 0);
          for (int j = 0; j < Ns_node; j++) {
            float z = Z_p[sp[j]];
            int b = (int)((z - lmin) * inv_r);
            if (b < 0) b = 0;
            if (b >= N_HIST_BINS) b = N_HIST_BINS - 1;
            cnt_hist[b]++;
            const float* gi = G + (size_t)sp[j] * K;
            const float* hi = H + (size_t)sp[j] * K;
            for (int k = 0; k < K; k++) {
              G_hist[b * K + k] += gi[k];
              H_hist[b * K + k] += hi[k];
            }
          }
          std::fill(G_Lp.begin(), G_Lp.end(), 0.0f);
          std::fill(H_Lp.begin(), H_Lp.end(), 0.0f);
          int n_left = 0;
          for (int b = 0; b < N_HIST_BINS - 1; b++) {
            n_left += cnt_hist[b];
            for (int k = 0; k < K; k++) {
              G_Lp[k] += G_hist[b * K + k];
              H_Lp[k] += H_hist[b * K + k];
            }
            int n_right = Ns_node - n_left;
            if (n_left < min_leaf || n_right < min_leaf) continue;
            float gain = 0.0f;
            for (int k = 0; k < K; k++) {
              float GR = G_tot[k] - G_Lp[k];
              float HR = H_tot[k] - H_Lp[k];
              gain += G_Lp[k] * G_Lp[k] / (H_Lp[k] + reg_lambda) +
                      GR * GR / (HR + reg_lambda);
            }
            gain = 0.5f * (gain - parent_score);
            float split_val = lmin + (b + 1) / inv_r;
            if (gain > tl_gain) {
              tl_gain = gain;
              tl_p = p;
              tl_thr = split_val;
            }
          }
        }
#pragma omp critical
        {
          if (tl_gain > best_gain) {
            best_gain = tl_gain;
            best_p = tl_p;
            best_thr = tl_thr;
          }
        }
      }
#else
      {
        std::vector<float> G_hist(N_HIST_BINS * K);
        std::vector<float> H_hist(N_HIST_BINS * K);
        std::vector<int> cnt_hist(N_HIST_BINS);
        std::vector<float> G_Lp(K), H_Lp(K);
        for (int p = 0; p < P; p++) {
          const float* Z_p = Z_sub + (size_t)p * Ns;
          float lmin = Z_p[sp[0]], lmax = Z_p[sp[0]];
          for (int j = 1; j < Ns_node; j++) {
            float v = Z_p[sp[j]];
            if (v < lmin) lmin = v;
            if (v > lmax) lmax = v;
          }
          float range = lmax - lmin;
          if (range < 1e-8f) continue;
          float inv_r = N_HIST_BINS / range;
          std::fill(G_hist.begin(), G_hist.end(), 0.0f);
          std::fill(H_hist.begin(), H_hist.end(), 0.0f);
          std::fill(cnt_hist.begin(), cnt_hist.end(), 0);
          for (int j = 0; j < Ns_node; j++) {
            float z = Z_p[sp[j]];
            int b = (int)((z - lmin) * inv_r);
            if (b < 0) b = 0;
            if (b >= N_HIST_BINS) b = N_HIST_BINS - 1;
            cnt_hist[b]++;
            const float* gi = G + (size_t)sp[j] * K;
            const float* hi = H + (size_t)sp[j] * K;
            for (int k = 0; k < K; k++) {
              G_hist[b * K + k] += gi[k];
              H_hist[b * K + k] += hi[k];
            }
          }
          std::fill(G_Lp.begin(), G_Lp.end(), 0.0f);
          std::fill(H_Lp.begin(), H_Lp.end(), 0.0f);
          int n_left = 0;
          for (int b = 0; b < N_HIST_BINS - 1; b++) {
            n_left += cnt_hist[b];
            for (int k = 0; k < K; k++) {
              G_Lp[k] += G_hist[b * K + k];
              H_Lp[k] += H_hist[b * K + k];
            }
            int n_right = Ns_node - n_left;
            if (n_left < min_leaf || n_right < min_leaf) continue;
            float gain = 0.0f;
            for (int k = 0; k < K; k++) {
              float GR = G_tot[k] - G_Lp[k];
              float HR = H_tot[k] - H_Lp[k];
              gain += G_Lp[k] * G_Lp[k] / (H_Lp[k] + reg_lambda) +
                      GR * GR / (HR + reg_lambda);
            }
            gain = 0.5f * (gain - parent_score);
            float split_val = lmin + (b + 1) / inv_r;
            if (gain > best_gain) {
              best_gain = gain;
              best_p = p;
              best_thr = split_val;
            }
          }
        }
      }
#endif

      if (best_p < 0) {
        tree->is_leaf[t] = 1;
        {
          std::vector<int> tmp;
          tmp.swap(samples);
        }
        continue;
      }
      tree->is_leaf[t] = 0;
      tree->split_hyp_idx[t] = best_p;
      tree->split_threshold[t] = best_thr;
      tree->split_gain[t] = best_gain;

      int t_left = 2 * t + 1, t_right = 2 * t + 2;
      const float* Z_best = Z_sub + (size_t)best_p * Ns;
      node_samples[t_left].reserve(n_in / 2 + 8);
      node_samples[t_right].reserve(n_in / 2 + 8);
      for (int idx : samples) {
        if (Z_best[idx] < best_thr)
          node_samples[t_left].push_back(idx);
        else
          node_samples[t_right].push_back(idx);
      }
      {
        std::vector<int> tmp;
        tmp.swap(samples);
      }
    }
  }
  return tree;
}

// ── Best-First Tree Growth
// ────────────────────────────────────────────────────

static bool evaluate_node_split(
    const float* const* hyp_caches, int P, const std::vector<int>& samples,
    const float* G_full, const float* H_full, int K, float reg_lambda,
    int min_leaf, int min_split, int max_depth, int current_depth,
    float& out_gain, int& out_best_p, float& out_best_thr,
    std::vector<float>& out_G_sum, std::vector<float>& out_H_sum) {
  int Ns_node = (int)samples.size();
  if (Ns_node < min_split || current_depth >= max_depth) {
    out_G_sum.assign(K, 0.0f);
    out_H_sum.assign(K, 0.0f);
    for (int idx : samples)
      for (int k = 0; k < K; ++k) {
        out_G_sum[k] += G_full[idx * K + k];
        out_H_sum[k] += H_full[idx * K + k];
      }
    out_gain = -1.0f;
    return false;
  }

  const int* sp = samples.data();
  std::vector<int> sub_buf;
  int Ns = Ns_node;
  if (Ns_node > 25000) {
    Ns = 25000;
    sub_buf.assign(samples.begin(), samples.begin() + Ns);
    sp = sub_buf.data();
  }

  std::vector<float> G_tot(K, 0.0f), H_tot(K, 0.0f);
  for (int j = 0; j < Ns; ++j) {
    int idx = sp[j];
    for (int k = 0; k < K; ++k) {
      G_tot[k] += G_full[idx * K + k];
      H_tot[k] += H_full[idx * K + k];
    }
  }
  float parent_score = 0.0f;
  for (int k = 0; k < K; ++k)
    parent_score += G_tot[k] * G_tot[k] / (H_tot[k] + reg_lambda);

  float best_gain = 1e-5f;
  int best_p = -1;
  float best_thr = 0.0f;

#pragma omp parallel
  {
    std::vector<float> G_hist(N_HIST_BINS * K);
    std::vector<float> H_hist(N_HIST_BINS * K);
    std::vector<int> cnt_hist(N_HIST_BINS);
    std::vector<float> G_L(K), H_L(K);
    float tl_gain = 1e-5f;
    int tl_p = -1;
    float tl_thr = 0.0f;
#pragma omp for nowait
    for (int p = 0; p < P; ++p) {
      const float* Z_p = hyp_caches[p];
      float lmin = Z_p[sp[0]], lmax = Z_p[sp[0]];
      for (int j = 1; j < Ns; ++j) {
        float v = Z_p[sp[j]];
        if (v < lmin) lmin = v;
        if (v > lmax) lmax = v;
      }
      float range = lmax - lmin;
      if (range < 1e-8f) continue;
      float inv_r = N_HIST_BINS / range;
      std::fill(G_hist.begin(), G_hist.end(), 0.0f);
      std::fill(H_hist.begin(), H_hist.end(), 0.0f);
      std::fill(cnt_hist.begin(), cnt_hist.end(), 0);
      for (int j = 0; j < Ns; ++j) {
        float z = Z_p[sp[j]];
        int b = (int)((z - lmin) * inv_r);
        b = std::max(0, std::min(N_HIST_BINS - 1, b));
        cnt_hist[b]++;
        const float* gi = G_full + (size_t)sp[j] * K;
        const float* hi = H_full + (size_t)sp[j] * K;
        for (int k = 0; k < K; ++k) {
          G_hist[b * K + k] += gi[k];
          H_hist[b * K + k] += hi[k];
        }
      }
      std::fill(G_L.begin(), G_L.end(), 0.0f);
      std::fill(H_L.begin(), H_L.end(), 0.0f);
      int n_left = 0;
      for (int b = 0; b < N_HIST_BINS - 1; ++b) {
        n_left += cnt_hist[b];
        for (int k = 0; k < K; ++k) {
          G_L[k] += G_hist[b * K + k];
          H_L[k] += H_hist[b * K + k];
        }
        int n_right = Ns - n_left;
        if (n_left < min_leaf || n_right < min_leaf) continue;
        float gain = 0.0f;
        for (int k = 0; k < K; ++k) {
          float GR = G_tot[k] - G_L[k];
          float HR = H_tot[k] - H_L[k];
          gain += G_L[k] * G_L[k] / (H_L[k] + reg_lambda) +
                  GR * GR / (HR + reg_lambda);
        }
        gain = 0.5f * (gain - parent_score);
        float split_val = lmin + (b + 1) / inv_r;
        if (gain > tl_gain) {
          tl_gain = gain;
          tl_p = p;
          tl_thr = split_val;
        }
      }
    }
#pragma omp critical
    {
      if (tl_gain > best_gain) {
        best_gain = tl_gain;
        best_p = tl_p;
        best_thr = tl_thr;
      }
    }
  }

  out_G_sum.assign(K, 0.0f);
  out_H_sum.assign(K, 0.0f);
  for (int idx : samples)
    for (int k = 0; k < K; ++k) {
      out_G_sum[k] += G_full[idx * K + k];
      out_H_sum[k] += H_full[idx * K + k];
    }

  if (best_p >= 0) {
    out_gain = best_gain;
    out_best_p = best_p;
    out_best_thr = best_thr;
    return true;
  }
  out_gain = -1.0f;
  return false;
}

BFSTree* bfs_build_best_first(const float* const* hyp_caches, int P,
                              const std::vector<int>& train_indices,
                              const float* G_full, const float* H_full, int K,
                              int max_depth, int min_split, int min_leaf,
                              float reg_lambda) {
  BFSTree* tree = new BFSTree();
  tree->max_depth = max_depth;
  tree->K = K;
  int total_nodes_max = (1 << (max_depth + 1)) - 1;
  tree->total_nodes = total_nodes_max;
  tree->split_hyp_idx.assign(total_nodes_max, -1);
  tree->split_threshold.assign(total_nodes_max, 0.0f);
  tree->split_gain.assign(total_nodes_max, 0.0f);
  tree->leaf_values.assign((size_t)total_nodes_max * K, 0.0f);
  tree->is_leaf.assign(total_nodes_max, 1);

  std::vector<std::vector<int>> node_samples(total_nodes_max);
  node_samples[0] = train_indices;

  struct PQItem {
    float gain;
    int node_idx;
    int best_p;
    float best_thr;
    bool operator<(const PQItem& o) const { return gain < o.gain; }
  };
  std::priority_queue<PQItem> pq;

  auto process_node = [&](int node_idx, int depth) {
    auto& samples = node_samples[node_idx];
    if (samples.empty()) {
      tree->is_leaf[node_idx] = 1;
      return;
    }
    float gain;
    int best_p;
    float best_thr;
    std::vector<float> G_sum, H_sum;
    bool can_split = evaluate_node_split(
        hyp_caches, P, samples, G_full, H_full, K, reg_lambda, min_leaf,
        min_split, max_depth, depth, gain, best_p, best_thr, G_sum, H_sum);
    if (can_split && gain > 0 && depth < max_depth) {
      pq.push({gain, node_idx, best_p, best_thr});
    } else {
      tree->is_leaf[node_idx] = 1;
      float* lv = &tree->leaf_values[(size_t)node_idx * K];
      for (int k = 0; k < K; ++k) lv[k] = -(G_sum[k] / (H_sum[k] + reg_lambda));
      samples.clear();
      samples.shrink_to_fit();
    }
  };

  process_node(0, 0);

  while (!pq.empty()) {
    PQItem item = pq.top();
    pq.pop();
    int t = item.node_idx;
    if (!tree->is_leaf[t]) continue;
    if (item.gain <= 1e-5f) break;

    int depth = 0;
    for (int i = t; i > 0; i = (i - 1) / 2) depth++;
    if (depth >= max_depth) {
      tree->is_leaf[t] = 1;
      node_samples[t].clear();
      continue;
    }

    tree->is_leaf[t] = 0;
    tree->split_hyp_idx[t] = item.best_p;
    tree->split_threshold[t] = item.best_thr;
    tree->split_gain[t] = item.gain;

    int t_left = 2 * t + 1, t_right = 2 * t + 2;
    const float* Z_best = hyp_caches[item.best_p];
    auto& parent_samples = node_samples[t];
    node_samples[t_left].reserve(parent_samples.size() / 2 + 8);
    node_samples[t_right].reserve(parent_samples.size() / 2 + 8);
    for (int idx : parent_samples) {
      if (Z_best[idx] < item.best_thr)
        node_samples[t_left].push_back(idx);
      else
        node_samples[t_right].push_back(idx);
    }
    parent_samples.clear();
    parent_samples.shrink_to_fit();
    process_node(t_left, depth + 1);
    process_node(t_right, depth + 1);
  }

  for (int t = 0; t < total_nodes_max; ++t) {
    if (tree->is_leaf[t] && !node_samples[t].empty()) {
      std::vector<float> G_sum(K, 0.0f), H_sum(K, 0.0f);
      for (int idx : node_samples[t])
        for (int k = 0; k < K; ++k) {
          G_sum[k] += G_full[idx * K + k];
          H_sum[k] += H_full[idx * K + k];
        }
      float* lv = &tree->leaf_values[(size_t)t * K];
      for (int k = 0; k < K; ++k) lv[k] = -(G_sum[k] / (H_sum[k] + reg_lambda));
      node_samples[t].clear();
    }
  }
  return tree;
}

// ─────────────────────────────────────────────────────────────────────────────
// C API — BFSTree
// ─────────────────────────────────────────────────────────────────────────────
extern "C" {

void* bfstree_build(const float* Z, const float* thresholds, const float* G,
                    const float* H, int P, int N, int K, int max_depth,
                    int min_samples_split, int min_samples_leaf,
                    float reg_lambda) {
  BFSTree* tree = new BFSTree();
  tree->max_depth = max_depth;
  tree->K = K;
  int total_nodes = (1 << (max_depth + 1)) - 1;
  tree->total_nodes = total_nodes;
  tree->split_hyp_idx.assign(total_nodes, -1);
  tree->split_threshold.assign(total_nodes, 0.0f);
  tree->leaf_values.assign((size_t)total_nodes * K, 0.0f);
  tree->is_leaf.assign(total_nodes, 1);
  tree->split_gain.assign(total_nodes, 0.0f);

  std::vector<std::vector<int>> node_samples(total_nodes);
  node_samples[0].resize(N);
  std::iota(node_samples[0].begin(), node_samples[0].end(), 0);

  std::vector<float> G_sum(K), H_sum(K), G_tot(K), H_tot(K), G_L(K), H_L(K);

  for (int depth = 0; depth <= max_depth; depth++) {
    int n_nodes = 1 << depth, base = n_nodes - 1;
    bool is_last = (depth == max_depth);
    for (int local = 0; local < n_nodes; local++) {
      int t = base + local;
      auto& samples = node_samples[t];
      int n_in = (int)samples.size();
      if (n_in == 0) {
        tree->is_leaf[t] = 1;
        continue;
      }

      std::fill(G_sum.begin(), G_sum.end(), 0.0f);
      std::fill(H_sum.begin(), H_sum.end(), 0.0f);
      for (int idx : samples)
        for (int k = 0; k < K; k++) {
          G_sum[k] += G[idx * K + k];
          H_sum[k] += H[idx * K + k];
        }
      float* lv = &tree->leaf_values[(size_t)t * K];
      for (int k = 0; k < K; k++) lv[k] = -(G_sum[k] / (H_sum[k] + reg_lambda));

      if (is_last || n_in < min_samples_split) {
        tree->is_leaf[t] = 1;
        {
          std::vector<int> tmp;
          tmp.swap(samples);
        }
        continue;
      }

      const int* sp = samples.data();
      int Ns = n_in;
      std::vector<int> sub_buf;
      if (n_in > 25000) {
        Ns = 25000;
        sub_buf.assign(samples.begin(), samples.begin() + Ns);
        sp = sub_buf.data();
      }

      std::fill(G_tot.begin(), G_tot.end(), 0.0f);
      std::fill(H_tot.begin(), H_tot.end(), 0.0f);
      for (int j = 0; j < Ns; j++)
        for (int k = 0; k < K; k++) {
          G_tot[k] += G[sp[j] * K + k];
          H_tot[k] += H[sp[j] * K + k];
        }
      float parent_score = 0.0f;
      for (int k = 0; k < K; k++)
        parent_score += G_tot[k] * G_tot[k] / (H_tot[k] + reg_lambda);

      float best_gain = 1e-5f;
      int best_p = -1;
      float best_thr = 0.0f;
#ifdef _OPENMP
#pragma omp parallel
      {
        std::vector<float> z_buf(Ns), G_L_tl(K), H_L_tl(K);
        float tl_gain = 1e-5f;
        int tl_p = -1;
        float tl_thr = 0.0f;
#pragma omp for nowait
        for (int p = 0; p < P; p++) {
          const float* Z_p = Z + (size_t)p * N;
          for (int j = 0; j < Ns; j++) z_buf[j] = Z_p[sp[j]];
          for (int q = 0; q < 9; q++) {
            float thr = thresholds[q * P + p];
            std::fill(G_L_tl.begin(), G_L_tl.end(), 0.0f);
            std::fill(H_L_tl.begin(), H_L_tl.end(), 0.0f);
            int n_left = 0;
            for (int j = 0; j < Ns; j++) {
              if (z_buf[j] < thr) {
                ++n_left;
                for (int k = 0; k < K; k++) {
                  G_L_tl[k] += G[sp[j] * K + k];
                  H_L_tl[k] += H[sp[j] * K + k];
                }
              }
            }
            int n_right = Ns - n_left;
            if (n_left < min_samples_leaf || n_right < min_samples_leaf)
              continue;
            float gain = 0.0f;
            for (int k = 0; k < K; k++) {
              float GR = G_tot[k] - G_L_tl[k], HR = H_tot[k] - H_L_tl[k];
              gain += G_L_tl[k] * G_L_tl[k] / (H_L_tl[k] + reg_lambda) +
                      GR * GR / (HR + reg_lambda);
            }
            gain = 0.5f * (gain - parent_score);
            if (gain > tl_gain) {
              tl_gain = gain;
              tl_p = p;
              tl_thr = thr;
            }
          }
        }
#pragma omp critical
        {
          if (tl_gain > best_gain) {
            best_gain = tl_gain;
            best_p = tl_p;
            best_thr = tl_thr;
          }
        }
      }
#else
      {
        std::vector<float> z_buf(Ns);
        for (int p = 0; p < P; p++) {
          const float* Z_p = Z + (size_t)p * N;
          for (int j = 0; j < Ns; j++) z_buf[j] = Z_p[sp[j]];
          for (int q = 0; q < 9; q++) {
            float thr = thresholds[q * P + p];
            std::fill(G_L.begin(), G_L.end(), 0.0f);
            std::fill(H_L.begin(), H_L.end(), 0.0f);
            int n_left = 0;
            for (int j = 0; j < Ns; j++) {
              if (z_buf[j] < thr) {
                ++n_left;
                for (int k = 0; k < K; k++) {
                  G_L[k] += G[sp[j] * K + k];
                  H_L[k] += H[sp[j] * K + k];
                }
              }
            }
            int n_right = Ns - n_left;
            if (n_left < min_samples_leaf || n_right < min_samples_leaf)
              continue;
            float gain = 0.0f;
            for (int k = 0; k < K; k++) {
              float GR = G_tot[k] - G_L[k], HR = H_tot[k] - H_L[k];
              gain += G_L[k] * G_L[k] / (H_L[k] + reg_lambda) +
                      GR * GR / (HR + reg_lambda);
            }
            gain = 0.5f * (gain - parent_score);
            if (gain > best_gain) {
              best_gain = gain;
              best_p = p;
              best_thr = thr;
            }
          }
        }
      }
#endif
      if (best_p < 0) {
        tree->is_leaf[t] = 1;
        {
          std::vector<int> tmp;
          tmp.swap(samples);
        }
        continue;
      }
      tree->is_leaf[t] = 0;
      tree->split_hyp_idx[t] = best_p;
      tree->split_threshold[t] = best_thr;
      tree->split_gain[t] = best_gain;

      int t_left = 2 * t + 1, t_right = 2 * t + 2;
      const float* Z_best = Z + (size_t)best_p * N;
      node_samples[t_left].reserve(n_in / 2 + 8);
      node_samples[t_right].reserve(n_in / 2 + 8);
      for (int idx : samples) {
        if (Z_best[idx] < best_thr)
          node_samples[t_left].push_back(idx);
        else
          node_samples[t_right].push_back(idx);
      }
      {
        std::vector<int> tmp;
        tmp.swap(samples);
      }
    }
  }
  return static_cast<void*>(tree);
}

void bfstree_predict(void* handle, const float* Z_pred, int N_pred,
                     float* out) {
  const BFSTree* tree = static_cast<const BFSTree*>(handle);
  int K = tree->K;
  std::vector<int> node_assign(N_pred, 0);
  for (int depth = 0; depth < tree->max_depth; depth++) {
    int base = (1 << depth) - 1;
    for (int local = 0; local < (1 << depth); local++) {
      int t = base + local;
      if (tree->is_leaf[t]) continue;
      int p = tree->split_hyp_idx[t];
      float thr = tree->split_threshold[t];
      const float* Zp = Z_pred + (size_t)p * N_pred;
      int t_left = 2 * t + 1, t_right = 2 * t + 2;
      for (int i = 0; i < N_pred; i++)
        if (node_assign[i] == t)
          node_assign[i] = (Zp[i] < thr) ? t_left : t_right;
    }
  }
  for (int i = 0; i < N_pred; i++) {
    const float* lv = &tree->leaf_values[(size_t)node_assign[i] * K];
    float* oi = out + (size_t)i * K;
    for (int k = 0; k < K; k++) oi[k] = lv[k];
  }
}

void bfstree_free(void* handle) { delete static_cast<BFSTree*>(handle); }
int bfstree_get_K(void* handle) { return static_cast<BFSTree*>(handle)->K; }
int bfstree_get_max_depth(void* handle) {
  return static_cast<BFSTree*>(handle)->max_depth;
}
int bfstree_get_total_nodes(void* handle) {
  return static_cast<BFSTree*>(handle)->total_nodes;
}

void bfstree_get_split_indices(void* handle, int* out) {
  const BFSTree* tree = static_cast<const BFSTree*>(handle);
  for (int i = 0; i < tree->total_nodes; ++i) out[i] = tree->split_hyp_idx[i];
}

void bfstree_export(void* handle, int* split_hyp_idx, float* split_threshold,
                    float* leaf_values, uint8_t* is_leaf) {
  const BFSTree* tree = static_cast<const BFSTree*>(handle);
  int n = tree->total_nodes, K = tree->K;
  for (int i = 0; i < n; ++i) split_hyp_idx[i] = tree->split_hyp_idx[i];
  for (int i = 0; i < n; ++i) split_threshold[i] = tree->split_threshold[i];
  for (int i = 0; i < n * K; ++i) leaf_values[i] = tree->leaf_values[i];
  for (int i = 0; i < n; ++i) is_leaf[i] = tree->is_leaf[i];
}

void* bfstree_from_arrays(const int* split_hyp_idx,
                          const float* split_threshold,
                          const float* leaf_values, const uint8_t* is_leaf,
                          int total_nodes, int K, int max_depth) {
  BFSTree* tree = new BFSTree();
  tree->total_nodes = total_nodes;
  tree->K = K;
  tree->max_depth = max_depth;
  tree->split_hyp_idx.assign(split_hyp_idx, split_hyp_idx + total_nodes);
  tree->split_threshold.assign(split_threshold, split_threshold + total_nodes);
  tree->leaf_values.assign(leaf_values, leaf_values + (size_t)total_nodes * K);
  tree->is_leaf.assign(is_leaf, is_leaf + total_nodes);
  return static_cast<void*>(tree);
}

int bfstree_get_D(void* handle) {
  const BFSTree* tree = static_cast<const BFSTree*>(handle);
  return tree->D;
}

void bfstree_get_split_weights(void* handle, float* out_weights) {
  const BFSTree* tree = static_cast<const BFSTree*>(handle);
  if (tree->split_weights.size() > 0) {
    std::memcpy(out_weights, tree->split_weights.data(), tree->split_weights.size() * sizeof(float));
  }
}

void bfstree_set_split_weights(void* handle, int D, const float* weights) {
  BFSTree* tree = static_cast<BFSTree*>(handle);
  tree->D = D;
  tree->split_weights.assign(weights, weights + (size_t)tree->total_nodes * D);
}

}  // extern "C"
