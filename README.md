# CodeToAGI — Machine Learning From Scratch

**Build Real Machine Learning Models From The Ground Up**

This repository contains all code, notebooks, and challenge solutions for the **CodeToAGI Machine Learning Series** on YouTube.

---

## 📺 Episode 79: Regularization Explained (Ridge, Lasso & ElasticNet)

**"Your Model Is Lying To You — Here's How Regularization Fixes It"**

### What You’ll Learn
- Why models overfit and how to detect it
- The mathematics behind regularization (L1 vs L2 penalty)
- **Ridge Regression** (L2) — smooth coefficient shrinkage
- **Lasso Regression** (L1) — automatic feature selection (sets useless coefficients to exactly 0)
- **ElasticNet** — best of both worlds
- The importance of feature standardization before applying regularization
- How to choose between Ridge, Lasso, and ElasticNet in real projects

### Files in this Episode

| File                            | Description |
|-------------------------------|-----------|
| `ep79_challenge_solution.py`  | Complete solution to the challenge task |
| `manim_ep79.py`               | Manim animations used in the video |
| `generate_ep79.py`            | Video generation script |

---

### 🧩 Challenge Task (Episode 79)

**"Catch the Fake Feature"**

You are given data for 8 houses with **three features**:
- `sqft`
- `bedrooms`
- `noise` ← completely random & irrelevant feature

**Your Task:**
1. Fit a plain `LinearRegression` and observe the coefficient for the noise feature.
2. Standardize the features, then train **Ridge**, **Lasso**, and **ElasticNet**.
3. Which model drives the noise coefficient to **exactly zero**?

**Solution:** Check `ep79_challenge_solution.py`

---

## 📂 Repository Structure
