#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>

// OQTree — fitted oblique tree (sparse heap layout, serializable).

struct OQTree {
  int max_depth   = 0;
  int K           = 0;
  int D           = 0;
  int total_nodes = 0;
  std::vector<int>     split_hyp_idx;
  std::vector<float>   split_threshold;
  std::vector<float>   leaf_values;
  std::vector<uint8_t> is_leaf;
  std::vector<float>   split_gain;
  std::vector<float>   split_weights;  // [total_nodes × D]

  int D_num = 0;
  std::vector<float> na_means;  // [D]: numeric μ_f impute; cat cols: NaN-category rank
  std::vector<std::unordered_map<int, float>> cat_ranks;  // [D_cat]: raw value → rank
};
