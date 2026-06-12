import numpy as np

def main():
    print("Testing analytical coordinate descent on J(w)...")
    
    # Generate random A (symmetric positive definite) and G
    D = 10
    rng = np.random.default_rng(42)
    X = rng.normal(size=(100, D))
    A = X.T @ X + np.eye(D) # Hessian-weighted Gram matrix + lambda*I
    G = rng.normal(size=D) # gradient-weighted feature sum
    
    # Analytical WLS direction
    w_wls = np.linalg.solve(A, G)
    w_wls_norm = w_wls / np.linalg.norm(w_wls)
    
    # Let's run coordinate descent on J(w) = (w^T G)^2 / (w^T A w)
    # Start with a random direction
    w = rng.normal(size=D)
    w = w / np.linalg.norm(w)
    
    # Initial state
    w_G = np.dot(w, G)
    Aw = A @ w
    w_Aw = np.dot(w, Aw)
    
    print(f"Initial J(w): {w_G**2 / w_Aw:.6f}")
    
    for epoch in range(10):
        for f in range(D):
            # Parameters for 1D optimization along coordinate f
            a = w_G
            g = G[f]
            b = w_Aw
            c = Aw[f]
            d = A[f, f]
            
            # Optimal delta = (a*c - g*b) / (g*c - a*d)
            denom = g * c - a * d
            if abs(denom) > 1e-9:
                delta = (a * c - g * b) / denom
            else:
                delta = 0.0
                
            # Update w_f
            w[f] += delta
            
            # Update state variables in O(D)
            w_G += delta * G[f]
            # Update Aw: Aw_new = A @ (w + delta * e_f) = Aw + delta * A[:, f]
            Aw += delta * A[:, f]
            # Update w_Aw
            w_Aw = w_Aw + 2 * delta * c + (delta**2) * d
            
        J_val = w_G**2 / w_Aw
        cos_sim = abs(np.dot(w, w_wls_norm)) / np.linalg.norm(w)
        print(f"Epoch {epoch+1:2d} | J(w): {J_val:.6f} | Cosine to WLS: {cos_sim:.6f}")

if __name__ == "__main__":
    main()
