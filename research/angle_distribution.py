"""
Analyze child direction angle distribution relative to parent.
Runs a mathematical simulation of high-dimensional angles and
extracts winning child angles from OQBoostResearch training.
"""
from __future__ import annotations

import os
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from research.oqboost_research import OQBoostResearch


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle in degrees between two vectors."""
    cos = np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1.0, 1.0)
    return float(np.degrees(np.arccos(abs(cos))))


def run_math_simulation():
    print("\n==================================================")
    print("1. Mathematical Simulation: Angles of Random Vectors in d-Dimensions")
    print("==================================================")
    print("If a vector r is chosen completely randomly (Gaussian) in a d-dimensional subspace,")
    print("what is its angle relative to a fixed unit vector wp?")
    print("\n| Dimension d | Expected |cos(theta)| | Mean Angle (degrees) | Std Dev Angle |")
    print("|---|---|---|---|")
    
    rng = np.random.default_rng(42)
    dims = [2, 4, 8, 16, 30, 50, 100]
    
    for d in dims:
        # fixed unit vector
        wp = np.zeros(d)
        wp[0] = 1.0
        
        # draw 10,000 random vectors
        r = rng.normal(size=(10000, d))
        r_norms = np.linalg.norm(r, axis=1, keepdims=True)
        r_unit = r / (r_norms + 1e-12)
        
        # cosines and angles
        cosines = np.abs(np.dot(r_unit, wp))
        angles = np.degrees(np.arccos(cosines))
        
        print(f"| {d:3d} | {np.mean(cosines):.4f} | {np.mean(angles):.1f}° | {np.std(angles):.2f}° |")


def run_model_analysis():
    print("\n==================================================")
    print("2. Empirical Analysis: Winning Child Angles in OQBoostResearch")
    print("==================================================")
    
    # Generate a small rotated synthetic dataset for quick training
    print("Generating dataset (2000 samples, 30 features)...")
    base_X, base_y = make_classification(
        n_samples=2000,
        n_features=30,
        n_informative=10,
        random_state=42,
    )
    X = StandardScaler().fit_transform(base_X).astype(np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, base_y, test_size=0.25, random_state=42)
    
    strategies = ["random", "full"]
    
    for strategy in strategies:
        print(f"\nTraining model with '{strategy}' strategy...")
        clf = OQBoostResearch(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            use_wls=True,
            inherit_mode='orth',
            orth_strategy=strategy,
            random_state=42,
        )
        clf.fit(X_tr, y_tr)
        
        # Collect angles of winning inherit_O candidates
        records = clf.all_split_records()
        angles = []
        for r in records:
            if r.winner_type == 'inherit_O' and not np.isnan(r.angle_from_parent):
                angles.append(r.angle_from_parent)
                
        n_wins = len(angles)
        total_splits = len(records)
        win_rate = n_wins / max(total_splits, 1) * 100
        
        if n_wins > 0:
            mean_ang = np.mean(angles)
            std_ang = np.std(angles)
            min_ang = np.min(angles)
            max_ang = np.max(angles)
            print(f"  -> 'inherit_O' won {n_wins}/{total_splits} splits ({win_rate:.1f}% win rate)")
            print(f"  -> Winning Angles: Mean = {mean_ang:.2f}°, Std = {std_ang:.2f}°, Range = [{min_ang:.1f}°, {max_ang:.1f}°]")
        else:
            print(f"  -> 'inherit_O' did not win any splits (0/{total_splits})")


if __name__ == "__main__":
    run_math_simulation()
    run_model_analysis()
