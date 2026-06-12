import os
import torch
import numpy as np
import scipy.stats as stats
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler

# Simple mockup of split gain evaluation
def get_split_gain(proj, g, h, reg_lambda):
    # Sort samples by projection
    order = np.argsort(proj)
    proj_s = proj[order]
    g_s = g[order]
    h_s = h[order]
    
    # Cumulative sums
    g_left = np.cumsum(g_s)
    h_left = np.cumsum(h_s)
    g_total = g_left[-1]
    h_total = h_left[-1]
    
    g_right = g_total - g_left
    h_right = h_total - h_left
    
    # Calculate gain for all split points
    # Avoid empty splits
    n = len(proj)
    denom_left = h_left + reg_lambda
    denom_right = h_right + reg_lambda
    denom_total = h_total + reg_lambda
    
    gain = 0.5 * ( (g_left**2 / denom_left) + (g_right**2 / denom_right) - (g_total**2 / denom_total) )
    # Exclude empty splits
    gain[h_left == 0] = -1e9
    gain[h_right == 0] = -1e9
    
    return np.max(gain)

def main():
    print("Generating synthetic node data...")
    # Generate 500 samples, 20 features
    X, y = make_classification(n_samples=500, n_features=20, n_informative=10, random_state=42)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    
    # Generate dummy gradients and hessians
    rng = np.random.default_rng(42)
    p = 1.0 / (1.0 + np.exp(-X[:, 0] - X[:, 1] + rng.normal(0, 0.5, size=500)))
    y = (rng.uniform(size=500) < p).astype(np.float32)
    
    # gradients = p - y, hessians = p * (1 - p)
    g = p - y
    h = p * (1 - p) + 0.1 # add small value to keep positive
    reg_lambda = 1.0
    
    # Calculate WLS direction
    # A = X.T @ diag(h) @ X + lambda * I
    # b = - X.T @ g
    A = X.T @ (h[:, None] * X) + reg_lambda * np.eye(20)
    b = - X.T @ g
    w_wls = np.linalg.solve(A, b)
    
    # Normalize w_wls
    w_wls_norm = w_wls / np.linalg.norm(w_wls)
    
    # Generate 1000 random directions
    n_dirs = 1000
    W_rand = rng.normal(size=(n_dirs, 20))
    W_rand = W_rand / np.linalg.norm(W_rand, axis=1, keepdims=True)
    
    actual_gains = []
    proxy_linear_gain = []
    proxy_cos_wls = []
    proxy_cov_g = []
    proxy_cos_a = []
    
    print("Evaluating 1000 candidate directions...")
    for i in range(n_dirs):
        w = W_rand[i]
        proj = X @ w
        
        # 1. Actual split gain
        actual_gain = get_split_gain(proj, g, h, reg_lambda)
        actual_gains.append(actual_gain)
        
        # 2. Linear Gain proxy: J(w) = (b^T w)^2 / (w^T A w)
        num = np.dot(b, w) ** 2
        den = np.dot(w, A @ w)
        proxy_linear_gain.append(num / den)
        
        # 3. Cosine similarity with WLS
        cos_wls = abs(np.dot(w, w_wls_norm))
        proxy_cos_wls.append(cos_wls)
        
        # 4. Simple covariance with gradient: |sum(g_i * proj_i)|
        cov_g = abs(np.sum(g * proj))
        proxy_cov_g.append(cov_g)
        
        # 5. A-norm cosine similarity: cos_A(w, w_wls) = (w^T A w_wls) / sqrt((w^T A w)(w_wls^T A w_wls))
        num_a = np.dot(w, A @ w_wls)
        den_a = np.sqrt(np.dot(w, A @ w) * np.dot(w_wls, A @ w_wls))
        proxy_cos_a.append(abs(num_a / den_a))
        
    actual_gains = np.array(actual_gains)
    proxy_linear_gain = np.array(proxy_linear_gain)
    proxy_cos_wls = np.array(proxy_cos_wls)
    proxy_cov_g = np.array(proxy_cov_g)
    proxy_cos_a = np.array(proxy_cos_a)
    
    # Compute correlations
    print("\n--- Correlation with Actual Split Gain ---")
    for name, proxy in [
        ("Linear Gain Proxy J(w) = (b'w)^2 / (w'Aw)", proxy_linear_gain),
        ("A-norm Cosine to WLS", proxy_cos_a),
        ("Standard Cosine to WLS", proxy_cos_wls),
        ("Covariance with gradient |sum(g_i * proj_i)|", proxy_cov_g),
    ]:
        pearson, _ = stats.pearsonr(actual_gains, proxy)
        spearman, _ = stats.spearmanr(actual_gains, proxy)
        print(f"{name:50s} | Pearson: {pearson:.4f} | Spearman: {spearman:.4f}")

if __name__ == "__main__":
    main()
