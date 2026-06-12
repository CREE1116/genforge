import sys
import os
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import research.oqboost_research as oq_res

def main():
    print("Testing import namespaces...")
    print(f"oq_res: {oq_res}")
    print(f"oq_res._coordinate_descent_j: {oq_res._coordinate_descent_j}")
    
    # Generate data
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randint(0, 2, size=100)
    
    clf = oq_res.OQBoostResearch(
        n_estimators=1,
        learning_rate=0.1,
        max_depth=2,
        use_wls=True,
        inherit_mode="proxy_cd_1",
        device='cpu'
    )
    
    called = []
    orig = oq_res._coordinate_descent_j
    def patched(*args, **kwargs):
        called.append(True)
        return orig(*args, **kwargs)
        
    oq_res._coordinate_descent_j = patched
    
    clf.fit(X, y)
    print(f"Called? {called}")

if __name__ == "__main__":
    main()
