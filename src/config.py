"""
config.py — Project-wide configuration constants.

Contains ONLY genuine constants (business rules, fixed cost parameters,
sensitivity grid definitions, paths). Values that are DERIVED from data
(p, c, s, r_hat, k_hat, pG, pN, gN) live in parameter_estimation.py and
are computed per category at runtime.

Import what you need:
    from config import RESALABLE_REASONS, COST_PARAMS, DATA_DIR

────────────────────────────────────────────────────────────────────
Origin map (Colab "c_alıs_ır_son_hali_s_u_anlık.py"):
  RESALABLE_REASONS       → line 150
  COST_PARAMS (s_pct,g,d) → line 1043
  MC_SETTINGS             → lines 1218-1219
  SENSITIVITY_GRIDS       → lines 586-587 (k, g grid), line 617 (r grid)
  ORDER_STATUS_FILTER     → NEW: was line 167 ("Completed" only) —
                                   we extend to include "Returned" per
                                   data_analysis.py convention.
────────────────────────────────────────────────────────────────────
"""

from pathlib import Path
import numpy as np


# ════════════════════════════════════════════════════════════════════
# 1. PATHS
# ════════════════════════════════════════════════════════════════════
# Path(__file__) resolves to this config.py file.
# .parent       → src/
# .parent.parent → model1/   (project root)
PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR    = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
ARCHIVE_DIR = PROJECT_ROOT / "archive"
VISUALS_DIR = PROJECT_ROOT / "visuals"

TABLES_DIR  = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# Cleaned dataset produced by data_analysis.py
# (NOTE: data_analysis.py currently reads from master_data.csv but does
#  not persist the cleaned output. If you ever save `clean` to disk,
#  point to it here.)
CLEAN_DATA_PATH = DATA_DIR / "main_clean.csv"

# Ensure output directories exist (no-op if already present)
for _d in (RESULTS_DIR, TABLES_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ════════════════════════════════════════════════════════════════════
# 2. RETURN REASON CLASSIFICATION  (BUSINESS RULE)
# ════════════════════════════════════════════════════════════════════
# A returned unit is "resalable" if the return reason is non-defect-
# related (the unit comes back in saleable condition). Defect-related
# returns are written off.
#
# Data exposes four reason categories (per main_clean.csv):
#     Customer Preference  → resalable     (change of mind, size, color)
#     Wrong Item           → resalable     (item is fine, ordering error)
#     Defective            → non-resalable (manufacturing defect)
#     Damaged in Shipping  → non-resalable (transit damage)
#
# This split is a modeling choice, not a data field; revisiting it
# requires re-running estimate_parameters() to refresh k_hat.
# Colab origin: line 150
RESALABLE_REASONS = {"Customer Preference", "Wrong Item"}


# ════════════════════════════════════════════════════════════════════
# 3. ORDER STATUS FILTER  (DEMAND POPULATION)
# ════════════════════════════════════════════════════════════════════
# Which order_status values count as "fulfilled" demand?
#   - Completed: delivered and not returned                         ✓
#   - Returned:  delivered, then returned (still hit Gt population) ✓
#   - Cancelled: never shipped                                       ✗
#
# Colab originally filtered only "Completed" (line 167), missing 32
# rows. data_analysis.py uses the correct set; we follow that here.
ORDER_STATUS_FILTER = ["Completed", "Returned"]


# ════════════════════════════════════════════════════════════════════
# 4. COST PARAMETERS  (SHARED ACROSS CATEGORIES)
# ════════════════════════════════════════════════════════════════════
# Baseline values applied to every category UNLESS the caller passes
# explicit base_params in main().
#
# Per-category p (selling price) and c (unit cost) are NOT here — they
# are computed from the data by parameter_estimation.compute_pc_per_cat().
# Salvage value s is DERIVED as s = c * s_pct (also computed at runtime).
#
# Colab origin: line 1043
#   shared = dict(s_pct=0.20, g=8.0, d=2.0)
COST_PARAMS = {
    "s_pct": 0.20,   # salvage fraction:  s = s_pct * c
    "g":     8.0,    # shortage penalty per unit of lost demand
    "d":     2.0,    # return collection cost per unit
}


# ════════════════════════════════════════════════════════════════════
# 5. MONTE CARLO SETTINGS
# ════════════════════════════════════════════════════════════════════
# Default replication counts. Override at call sites if memory allows.
# Colab origin:
#   J_baseline    = 200  (line 1218)
#   J_sensitivity = 100  (line 1219)
MC_SETTINGS = {
    "J_baseline":         200,
    "J_sensitivity":      100,
    "seed_baseline":      0,
    "seed_sensitivity":   1,
    "seed_return_grid":   2,
}

# ════════════════════════════════════════════════════════════════════
# 6. SENSITIVITY GRIDS  (extended per supervisor feedback #5)
# ════════════════════════════════════════════════════════════════════
# Dense grids covering both baseline regime and EXTREME values, to expose
# regime transitions in (a) Q* sensitivity, (b) SAA vs Bayesian gap,
# (c) sequencing gap S3-S1.
#
# Supervisor feedback #5: "different assumptions don't matter much — try
# other parameters for larger differences" → grids extended to extremes:
#   g up to 100 (was 20)  — large shortage penalty regime
#   r up to 0.30 (was 0.16) — high-return regime (some retail categories)
#   k full range 0.20-0.95 (kept)
#
# Colab origin: lines 586-587 (k, g grid), line 617 (r grid) — values
# extended and key names switched to plain "g","k","r" to match downstream
# consumers (sensitivity.py).
SENSITIVITY_GRIDS = {
    "g": [1.0, 3.0, 5.0, 8.0, 12.0, 20.0, 35.0, 50.0, 75.0, 100.0],   # 10
    "k": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.95],             # 8
    "r": [0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.25, 0.30],       # 9
}

# ════════════════════════════════════════════════════════════════════
# 7. BAYESIAN PRIOR HYPERPARAMETERS (weakly informative)
# ════════════════════════════════════════════════════════════════════
# Philosophy A (weakly informative): priors are deliberately weak so the
# posterior is DATA-DOMINATED even at small m. No exogenous demand info is
# injected → the comparison with SAA stays fair (no test-data leakage).
#
# DEMAND MODEL — Negative Binomial-Beta (Berk et al. 2007, Theorem 3.1):
#   Category demand is OVERDISPERSED (VMR ~ 3-6 >> 1; AIC selects NegBin
#   over Poisson). A pure Poisson-Gamma posterior predictive collapses to
#   VMR~1 and cannot reproduce this spread, so we model
#       G_t ~ NegBin(r_NB, p),  with r_NB KNOWN (moment-matched per
#       category/period), and  p ~ Beta(a0, b0).
#   By conjugacy the posterior is p ~ Beta(a0 + r_NB*m, b0 + sum G_t).
#   Berk et al. (2007), p.259, show the uniform Beta(1,1) prior yields a
#   PROPER predictive density.
#
# RETURN / RESALABILITY — Beta-Bernoulli (unchanged):
#   r, k ~ Beta(1,1); standard conjugate prior for Bernoulli/Binomial.
#
# General weakly-informative guidance: Gelman et al. (2013), BDA Ch. 2, 5.
# These are DEFAULTS; every function accepts overrides for prior-
# sensitivity analysis (Step 15).
BAYESIAN_PRIORS = {
    # NegBin-Beta on demand: p ~ Beta(a0, b0). r_NB is moment-matched
    # from data per category/period (not a prior — a fixed dispersion).
    "demand_p_a0": 1.0,
    "demand_p_b0": 1.0,
    # Beta-Bernoulli on return probability r:  r ~ Beta(a0, b0)
    "r_a0": 1.0,
    "r_b0": 1.0,
    # Beta-Bernoulli on resalability k:  k ~ Beta(a0, b0)
    "k_a0": 1.0,
    "k_b0": 1.0,
}