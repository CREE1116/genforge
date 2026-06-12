import torch
import time

def _node_score(G_sum: torch.Tensor, H_sum: torch.Tensor, lam: float) -> float:
    return float((G_sum.pow(2) / (H_sum + lam)).sum() / 2)

def _best_threshold_original(proj: torch.Tensor, G: torch.Tensor, H: torch.Tensor,
                             lam: float, n_bins: int = 64) -> tuple[float, float]:
    N = len(proj)
    order = proj.argsort()
    proj_s = proj[order]
    G_s = G[order]
    H_s = H[order]

    G_tot = G.sum(0)
    H_tot = H.sum(0)
    root = _node_score(G_tot, H_tot, lam)

    G_left = torch.zeros_like(G_tot)
    H_left = torch.zeros_like(H_tot)

    best_gain = -1e18
    best_t = float(proj_s[0])

    step = max(1, N // n_bins)
    for i in range(step, N - step, step):
        G_left = G_left + G_s[i - step:i].sum(0)
        H_left = H_left + H_s[i - step:i].sum(0)
        G_right = G_tot - G_left
        H_right = H_tot - H_left
        if float(H_left.sum()) < 0.1 or float(H_right.sum()) < 0.1:
            continue
        gain = (_node_score(G_left, H_left, lam) +
                _node_score(G_right, H_right, lam) - root)
        if gain > best_gain:
            best_gain = gain
            best_t = float((proj_s[i - 1] + proj_s[i]) / 2)

    return best_t, best_gain

def _best_threshold_vectorized(proj: torch.Tensor, G: torch.Tensor, H: torch.Tensor,
                               lam: float, n_bins: int = 64) -> tuple[float, float]:
    N = len(proj)
    order = proj.argsort()
    proj_s = proj[order]
    G_s = G[order]
    H_s = H[order]

    G_tot = G.sum(0)
    H_tot = H.sum(0)
    root = float((G_tot.pow(2) / (H_tot + lam)).sum() / 2)

    G_cum = torch.cumsum(G_s, dim=0)
    H_cum = torch.cumsum(H_s, dim=0)

    step = max(1, N // n_bins)
    range_t = torch.arange(step, N - step, step, dtype=torch.long, device=proj.device)
    if len(range_t) == 0:
        return float(proj_s[0]), -1e18

    idxs = range_t - 1
    G_left = G_cum[idxs]  # shape (n_bins - 1, K)
    H_left = H_cum[idxs]  # shape (n_bins - 1, K)

    G_right = G_tot.unsqueeze(0) - G_left
    H_right = H_tot.unsqueeze(0) - H_left

    score_left = (G_left.pow(2) / (H_left + lam)).sum(dim=-1) / 2
    score_right = (G_right.pow(2) / (H_right + lam)).sum(dim=-1) / 2

    gains = score_left + score_right - root

    # Mask valid splits
    valid_mask = (H_left.sum(dim=-1) >= 0.1) & (H_right.sum(dim=-1) >= 0.1)
    if not valid_mask.any():
        return float(proj_s[0]), -1e18

    gains = torch.where(valid_mask, gains, torch.tensor(-1e18, dtype=gains.dtype, device=gains.device))

    best_idx = gains.argmax()
    best_gain = float(gains[best_idx])
    
    i = int(range_t[best_idx])
    best_t = float((proj_s[i - 1] + proj_s[i]) / 2)

    return best_t, best_gain

def main():
    proj = torch.randn(1000)
    G = torch.randn(1000, 2)
    H = torch.rand(1000, 2) + 0.1
    lam = 1.0

    # Warmup
    _best_threshold_original(proj, G, H, lam)
    _best_threshold_vectorized(proj, G, H, lam)

    t0 = time.time()
    for _ in range(200):
        t1, g1 = _best_threshold_original(proj, G, H, lam)
    orig_time = time.time() - t0

    t0 = time.time()
    for _ in range(200):
        t2, g2 = _best_threshold_vectorized(proj, G, H, lam)
    vec_time = time.time() - t0

    print(f"Original threshold: {t1}, gain: {g1}")
    print(f"Vectorized threshold: {t2}, gain: {g2}")
    print(f"Difference: threshold diff = {abs(t1-t2)}, gain diff = {abs(g1-g2)}")
    print(f"Original time (200 runs): {orig_time:.4f}s")
    print(f"Vectorized time (200 runs): {vec_time:.4f}s")
    print(f"Speedup: {orig_time / vec_time:.1f}x")

if __name__ == "__main__":
    main()
