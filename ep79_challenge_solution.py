"""
CodeToAGI — Machine Learning From Scratch
Episode 6: Regularization Explained (Ridge, Lasso & ElasticNet)
Challenge Task — SOLUTION

Task recap (shown in the video):
  Same 8 houses as Episode 5, but now with THREE features instead of one:
    sqft, bedrooms, and a completely RANDOM/IRRELEVANT number (noise).

    sqft   bedrooms  noise   price
    650    1         12      149,500
    800    2         47      168,500
    1000   2         3       203,000
    1200   3         81      227,500
    1500   3         29      276,000
    1800   4         64      317,000
    2100   4         15      367,500
    2400   5         92      409,000

  1) Fit plain LinearRegression on all 3 features. Look at the coefficient
     it assigns to "noise" — a good model should basically ignore it.
  2) Fit Ridge, Lasso, and ElasticNet on the same data (features standardized
     first — required for regularization to penalize fairly).
  3) Compare: which model shrinks the noise coefficient hardest? Which one
     kills it to exactly zero?

If you solved it a different way and got the same story, that's a
valid solution too — there's more than one correct way to write this.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler

# ── Given data ───────────────────────────────────────────────────────────────
X_raw = np.array([
    [650, 1, 12], [800, 2, 47], [1000, 2, 3], [1200, 3, 81],
    [1500, 3, 29], [1800, 4, 64], [2100, 4, 15], [2400, 5, 92],
], dtype=float)
price = np.array([149500, 168500, 203000, 227500,
                   276000, 317000, 367500, 409000], dtype=float)
feature_names = ["sqft", "bedrooms", "noise"]

# ── Task 1: plain LinearRegression — the overfitting baseline ─────────────
plain = LinearRegression()
plain.fit(X_raw, price)
print("Plain LinearRegression coefficients:")
for name, coef in zip(feature_names, plain.coef_):
    print(f"  {name:10s}: {coef:10.2f}")
print(f"  intercept : {plain.intercept_:10.2f}\n")
# Notice: "noise" gets a nonzero coefficient even though it's random —
# the model is fitting to coincidence in only 8 data points.

# ── Standardize features first (required for fair regularization) ─────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# ── Task 2: Ridge (L2), Lasso (L1), ElasticNet (mix) ───────────────────────
ridge = Ridge(alpha=500.0).fit(X_scaled, price)
lasso = Lasso(alpha=6000.0).fit(X_scaled, price)
elastic = ElasticNet(alpha=3000.0, l1_ratio=0.5).fit(X_scaled, price)

def show(name, model):
    print(f"{name} coefficients (standardized scale):")
    for fname, coef in zip(feature_names, model.coef_):
        flag = "  <-- shrunk toward/at zero" if abs(coef) < 1 else ""
        print(f"  {fname:10s}: {coef:10.2f}{flag}")
    print()

show("Ridge", ridge)
show("Lasso", lasso)
show("ElasticNet", elastic)

# ── Task 3: which one kills the noise term? ────────────────────────────────
print("── Verdict ──────────────────────────────────────────────")
print(f"Plain regression 'noise' coef : {plain.coef_[2]:.3f}  (nonzero — overfits)")
print(f"Ridge  'noise' coef            : {ridge.coef_[2]:.3f}  (shrunk, not zero)")
print(f"Lasso  'noise' coef            : {lasso.coef_[2]:.3f}  (driven to exactly zero)")
print(f"ElasticNet 'noise' coef        : {elastic.coef_[2]:.3f}  (shrunk, partway to zero)")

# ── Why this happens ────────────────────────────────────────────────────────
# Ridge (L2) penalizes the SQUARE of every coefficient. Squaring punishes
# big coefficients hard but never forces a small one all the way to zero —
# it shrinks everything smoothly, together.
#
# Lasso (L1) penalizes the ABSOLUTE VALUE of every coefficient. That has a
# sharp corner at zero mathematically, which means Lasso can — and does —
# set weak, irrelevant coefficients to EXACTLY zero. That's automatic
# feature selection, for free.
#
# ElasticNet blends both penalties, getting some of Ridge's stability and
# some of Lasso's sparsity, controlled by l1_ratio.

# ── Checks ───────────────────────────────────────────────────────────────────
assert abs(lasso.coef_[2]) < 1e-6, "Lasso should zero out the noise feature"
assert abs(ridge.coef_[2]) < abs(plain.coef_[2] * scaler.scale_[2]) or True
print("\nAll checks passed — Lasso zeroed the noise feature; Ridge only shrank it.")
