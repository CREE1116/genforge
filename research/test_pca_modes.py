import os
import sys
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.oqboost_research import OQBoostResearch

def test_mode(mode):
    print(f"\n--- Testing mode: {mode} ---")
    digits = load_digits()
    X = StandardScaler().fit_transform(digits.data).astype(np.float32)
    y = digits.target
    
    clf = OQBoostResearch(
        n_estimators=10,
        learning_rate=0.1,
        max_depth=4,
        use_wls=True,
        inherit_mode=mode,
        inherited_rp_ratio=1.0,
        n_random=0,
        n_inherit=4,
        random_state=42,
        device='cpu',
    )
    clf.fit(X, y)
    print(f"Success! Number of trees trained: {len(clf.trees_)}")

if __name__ == "__main__":
    test_mode('random_only')
    test_mode('cache_pca')
    test_mode('cache_lineage_orth')
    test_mode('pobs')
    test_mode('qmc')
