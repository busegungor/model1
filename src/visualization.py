"""
visualization.py — Model-output figures for the thesis (Results section).

FOUR figures, two families:

  RESULTS (method comparison — the main contribution):
    FIGURE 1  fig_crossover     : relative regret vs sample size m_train,
              2x2 panels per period, SAA vs Bayesian, three g scenarios,
              +/-1 s.e. band. The crossover result.
    FIGURE 2  fig_sequencing    : gap_pct = (cost_S3-cost_S1)/cost_S1 bar
              chart (period x category). When does sequencing matter.

  DESCRIPTIVE-OF-MODEL (single full-data fit, salvaged & updated from the
  old Colab plots; benchmark reference replaced by Bayesian):
    FIGURE 3  plot_net_demand_distribution : net-demand histogram with the
              SAA and Bayesian optima marked.
    FIGURE 4  plot_cost_curves  : EP(Q) cost curve per category with SAA and
              Bayesian optima (replaces the old SAA-vs-benchmark version).

NOTE: the old r/k sensitivity plots are intentionally dropped — that
analysis was archived (r,k are data estimates, not swept).

Reads:  comparison_outputs/method_comparison.csv
        comparison_outputs/sequencing_importance.csv
Writes: comparison_outputs/fig_*.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

M_ORDER = ["5", "10", "15", "20", "30", "50", "75", "all"]
PERIODS = ["W", "2W", "3W", "4W"]
G_STYLES = {5.0: "dotted", 8.0: "solid", 20.0: "dashed"}
METHOD_COLOR = {"SAA": "#1f6feb", "Bayesian": "#e8730c"}


def _m_to_x(m_series):
    return m_series.map({m: i for i, m in enumerate(M_ORDER)})


def _alpha(p, c, s, g):
    cu = p + g - c; co = c - s
    return cu / (cu + co)


# ════════════════════════════════════════════════════════════════════
# FIGURE 1 — crossover (regret vs m_train), 2x2 periods, SAA vs Bayesian
# ════════════════════════════════════════════════════════════════════
def fig_crossover(df, out_path, g_values=(5.0, 8.0, 20.0)):
    agg = (df[df.g.isin(g_values)]
           .groupby(["period", "g", "m_train", "method"])
           .agg(regret=("regret_mean", "mean"),
                se=("regret_se", lambda x: np.sqrt(np.mean(x**2)) / np.sqrt(len(x))))
           .reset_index())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    axes = axes.flatten()
    for ax, period in zip(axes, PERIODS):
        sub = agg[agg.period == period]
        if sub.empty:
            ax.set_visible(False); continue
        for method in ("SAA", "Bayesian"):
            for g in g_values:
                d = sub[(sub.method == method) & (sub.g == g)]
                if d.empty:
                    continue
                d = d.set_index("m_train").reindex(M_ORDER).dropna(subset=["regret"]).reset_index()
                x = _m_to_x(d["m_train"]); y = d["regret"].values; se = d["se"].values
                ax.plot(x, y, color=METHOD_COLOR[method], linestyle=G_STYLES[g],
                        linewidth=1.8)
                ax.fill_between(x, y - se, y + se, color=METHOD_COLOR[method],
                                alpha=0.10, linewidth=0)
        ax.set_title(f"Period {period}", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(M_ORDER))); ax.set_xticklabels(M_ORDER)
        ax.axhline(0, color="0.7", linewidth=0.8, zorder=0)
        ax.grid(True, alpha=0.25); ax.set_ylabel("Relative regret")
    for ax in axes[2:]:
        ax.set_xlabel("Training sample size  m_train")

    legend_elems = [
        Line2D([0], [0], color=METHOD_COLOR["SAA"], lw=2, label="SAA (nonparametric)"),
        Line2D([0], [0], color=METHOD_COLOR["Bayesian"], lw=2, label="Bayesian (parametric)"),
        Line2D([0], [0], color="0.3", lw=2, linestyle="dotted", label="g = 5"),
        Line2D([0], [0], color="0.3", lw=2, linestyle="solid", label="g = 8"),
        Line2D([0], [0], color="0.3", lw=2, linestyle="dashed", label="g = 20"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=5,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Relative regret by sample size and period", fontsize=13, y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out_path}")


# ════════════════════════════════════════════════════════════════════
# FIGURE 1b/1c — generic: any metric (cost, Q) vs m_train, 2x2 periods
# ════════════════════════════════════════════════════════════════════
def fig_metric_vs_samplesize(df, out_path, value_col, se_col=None,
                             ylabel="", suptitle="", g_values=(5.0, 8.0, 20.0)):
    """
    Same layout as fig_crossover (2x2 periods, SAA vs Bayesian, 3 g styles,
    categories averaged) but for an arbitrary metric column — e.g. cost_mean
    or Q_mean. Lets us see how the raw cost and the order quantity Q move
    with sample size, as a method comparison, per period.
    """
    aggfns = {"val": (value_col, "mean")}
    if se_col and se_col in df.columns:
        aggfns["se"] = (se_col, lambda x: np.sqrt(np.mean(x**2)) / np.sqrt(len(x)))
    agg = (df[df.g.isin(g_values)]
           .groupby(["period", "g", "m_train", "method"])
           .agg(**aggfns).reset_index())
    has_se = "se" in agg.columns

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    axes = axes.flatten()
    for ax, period in zip(axes, PERIODS):
        sub = agg[agg.period == period]
        if sub.empty:
            ax.set_visible(False); continue
        for method in ("SAA", "Bayesian"):
            for g in g_values:
                d = sub[(sub.method == method) & (sub.g == g)]
                if d.empty:
                    continue
                d = d.set_index("m_train").reindex(M_ORDER).dropna(subset=["val"]).reset_index()
                x = _m_to_x(d["m_train"]); y = d["val"].values
                ax.plot(x, y, color=METHOD_COLOR[method], linestyle=G_STYLES[g],
                        linewidth=1.8)
                if has_se:
                    se = d["se"].values
                    ax.fill_between(x, y - se, y + se, color=METHOD_COLOR[method],
                                    alpha=0.10, linewidth=0)
        ax.set_title(f"Period {period}", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(M_ORDER))); ax.set_xticklabels(M_ORDER)
        ax.grid(True, alpha=0.25); ax.set_ylabel(ylabel)
    for ax in axes[2:]:
        ax.set_xlabel("Training sample size  m_train")

    legend_elems = [
        Line2D([0], [0], color=METHOD_COLOR["SAA"], lw=2, label="SAA (nonparametric)"),
        Line2D([0], [0], color=METHOD_COLOR["Bayesian"], lw=2, label="Bayesian (parametric)"),
        Line2D([0], [0], color="0.3", lw=2, linestyle="dotted", label="g = 5"),
        Line2D([0], [0], color="0.3", lw=2, linestyle="solid", label="g = 8"),
        Line2D([0], [0], color="0.3", lw=2, linestyle="dashed", label="g = 20"),
    ]
    fig.legend(handles=legend_elems, loc="upper center", ncol=5,
               frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(suptitle, fontsize=13, y=1.06)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out_path}")


# ════════════════════════════════════════════════════════════════════
# FIGURE 2 — sequencing importance (gap_pct), grouped bars
# ════════════════════════════════════════════════════════════════════
def fig_sequencing(seq, out_path):
    """
    Sequencing gap_pct = (cost_S3 - cost_S1)/cost_S1, per period panel,
    SAA vs Bayesian side by side for each category. Small bars in BOTH
    methods => sequencing is irrelevant in this return regime, and that
    conclusion does not depend on the demand model.
    """
    cats = sorted(seq.category.unique())
    has_method = "method" in seq.columns
    methods = ["SAA", "Bayesian"] if has_method else ["(single)"]
    mcolor = {"SAA": METHOD_COLOR["SAA"], "Bayesian": METHOD_COLOR["Bayesian"],
              "(single)": "#888780"}
    x = np.arange(len(cats))
    width = 0.8 / len(methods)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.flatten()
    for ax, period in zip(axes, PERIODS):
        sub = seq[seq.period == period]
        if sub.empty:
            ax.set_visible(False); continue
        for j, meth in enumerate(methods):
            d = (sub[sub.method == meth] if has_method else sub)
            d = d.set_index("category").reindex(cats)
            ax.bar(x + (j - (len(methods) - 1) / 2) * width,
                   d["gap_pct"].values, width, label=meth, color=mcolor[meth])
        ax.axhline(0, color="0.7", linewidth=0.8)
        ax.set_title(f"Period {period}", fontsize=12, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(r"Sequencing cost band (%)")
        ax.grid(True, axis="y", alpha=0.25)
    if has_method:
        axes[0].legend(frameon=False, fontsize=9, title="Method")
    fig.suptitle("Maximum impact of intraperiod sequencing — SAA vs Bayesian",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out_path}")


# ════════════════════════════════════════════════════════════════════
# FIGURE 2b — S1/S4/S3 cost levels side by side (where does S4 sit?)
# ════════════════════════════════════════════════════════════════════
def fig_sequencing_levels(seq, out_path, method=None):
    """
    Per-period panel; in each, the three rule costs (S1 best-case netting,
    S4 empirical order-of-arrival, S3 worst-case) shown side by side for
    every category. Lets the reader see directly whether S4 sits near S1
    (sequencing irrelevant) or near S3 (sequencing matters).

    If the CSV has a `method` column, pass method="SAA" or "Bayesian" to
    pick one; the figure title notes which. (Levels are not comparable
    across methods on the same axis — the demand models give different cost
    scales — so we draw one method per figure rather than overlaying.)
    """
    if "method" in seq.columns and method is not None:
        seq = seq[seq.method == method]
    mtag = f" — {method}" if method else ""
    cats = sorted(seq.category.unique())
    x = np.arange(len(cats)); width = 0.26
    rule_color = {"cost_S1": "#52B788", "cost_S4": "#1f6feb", "cost_S3": "#E76F51"}
    rule_label = {"cost_S1": "S1 (best / netting)",
                  "cost_S4": "S4 (empirical)",
                  "cost_S3": "S3 (worst)"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    axes = axes.flatten()
    for ax, period in zip(axes, PERIODS):
        sub = seq[seq.period == period].set_index("category").reindex(cats)
        vals = sub[["cost_S1", "cost_S4", "cost_S3"]]
        if sub.empty or vals.isna().all().all():
            ax.set_visible(False); continue
        for j, col in enumerate(["cost_S1", "cost_S4", "cost_S3"]):
            ax.bar(x + (j - 1) * width, sub[col].values, width,
                   label=rule_label[col], color=rule_color[col])
        ax.set_title(f"Period {period}", fontsize=12, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(cats, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Expected cost"); ax.grid(True, axis="y", alpha=0.25)
        vmin = np.nanmin(vals.values); vmax = np.nanmax(vals.values)
        if np.isfinite(vmin) and np.isfinite(vmax):
            pad = (vmax - vmin) * 0.5 + 0.5
            ax.set_ylim(max(0, vmin - pad), vmax + pad)
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    fig.suptitle(f"Sequencing cost levels — S1 / S4 / S3{mtag}", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out_path}")


# ════════════════════════════════════════════════════════════════════
# FIGURE 3 — net-demand distribution with SAA & Bayesian optima
# (salvaged from Colab; benchmark reference replaced by Bayesian)
# ════════════════════════════════════════════════════════════════════
def plot_net_demand_distribution(N_saa, N_bay, Q_saa, Q_bay,
                                 title="", save_path=None):
    """Histogram of SAA net demand with SAA and Bayesian optima marked."""
    N_flat = N_saa.flatten()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(N_flat, bins=50, density=True, alpha=0.55, color="steelblue",
            label="Net demand $N_t^{(j)}$ (SAA)")
    ax.axvline(Q_saa, color=METHOD_COLOR["SAA"], ls="--", lw=2,
               label=f"$Q^*$ SAA = {Q_saa:.1f}")
    ax.axvline(Q_bay, color=METHOD_COLOR["Bayesian"], ls="--", lw=2,
               label=f"$Q^*$ Bayesian = {Q_bay:.1f}")
    ax.set_xlabel("Net Demand"); ax.set_ylabel("Density")
    ax.set_title(f"Net demand with $Q^*$ — {title} (representative)")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150); print(f"Saved -> {save_path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
# FIGURE 4 — EP(Q) cost curves, SAA vs Bayesian optima (all categories)
# ════════════════════════════════════════════════════════════════════
def plot_cost_curves(all_cost_data: dict, save_path=None):
    """
    Multi-panel EP(Q) curves. all_cost_data:
      {cat: {N_flat, p, c, s, g, Q_saa, Q_bay}}
    EP(Q) computed on the SAA net-demand pool; both optima marked.
    """
    cats = list(all_cost_data.keys())
    n = len(cats); ncols = 3; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axf = axes.flatten()
    for i, cat in enumerate(cats):
        ax = axf[i]; d = all_cost_data[cat]
        N = d["N_flat"]; cu = d["p"] + d["g"] - d["c"]; co = d["c"] - d["s"]
        Qg = np.linspace(N.min() * 0.3, N.max() * 1.5, 400)
        EP = np.array([cu * np.mean(np.maximum(N - q, 0))
                       + co * np.mean(np.maximum(q - N, 0)) for q in Qg])
        def ep(q): return float(cu * np.mean(np.maximum(N - q, 0))
                                + co * np.mean(np.maximum(q - N, 0)))
        ax.plot(Qg, EP, color="#444", lw=2, label="EP(Q)")
        ax.axvline(d["Q_saa"], color=METHOD_COLOR["SAA"], ls="--", lw=1.8,
                   label=f"$Q^*_{{SAA}}$={d['Q_saa']:.0f}")
        ax.scatter(d["Q_saa"], ep(d["Q_saa"]), color=METHOD_COLOR["SAA"], s=60, zorder=5)
        ax.axvline(d["Q_bay"], color=METHOD_COLOR["Bayesian"], ls="--", lw=1.8,
                   label=f"$Q^*_{{Bay}}$={d['Q_bay']:.0f}")
        ax.scatter(d["Q_bay"], ep(d["Q_bay"]), color=METHOD_COLOR["Bayesian"], s=60, zorder=5)
        EPmin = EP.min()
        ax.set_ylim(EPmin * 0.92, EPmin * 3.0)
        Qctr = 0.5 * (d["Q_saa"] + d["Q_bay"])
        ax.set_xlim(Qctr - 12, Qctr + 12)
        ax.set_title(cat, fontsize=11, fontweight="bold")
        ax.set_xlabel("Order Quantity $Q$"); ax.set_ylabel("Expected Cost $EP(Q)$")
        ax.legend(fontsize=8, loc="upper right"); ax.grid(True, alpha=0.3, ls="--")
    for ax in axf[n:]:
        ax.set_visible(False)
    fig.suptitle("Expected cost $EP(Q)$ — SAA vs Bayesian optima (full data, W)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=170, bbox_inches="tight"); print(f"Saved -> {save_path}")
    plt.close()


# ════════════════════════════════════════════════════════════════════
def _build_fulldata_fits(period="W", rule="S4", J=200, seed=0):
    """Helper: full-data SAA & Bayesian fits per category for figs 3-4."""
    from data_analysis import order_items_clean, returns_clean
    from preprocessing import build_category_datasets
    from parameter_estimation import build_category_params
    from monte_carlo import (build_net_demand_matrix,
                             build_bayesian_net_demand_matrix)
    datasets = build_category_datasets(order_items_clean, returns_clean, period=period)
    params = build_category_params(datasets, order_items_clean)
    out = {}
    for cat, ds in datasets.items():
        prm = params[cat]
        c, s, p, g = prm["c"], prm["s"], prm["p"], prm["g"]
        a = _alpha(p, c, s, g)
        N_saa = build_net_demand_matrix(ds, prm["r_hat"], prm["k_hat"],
                                        rule=rule, J=J, seed=seed)
        N_bay = build_bayesian_net_demand_matrix(ds, rule=rule, J=J, seed=seed)
        out[cat] = {"N_saa": N_saa, "N_bay": N_bay, "N_flat": N_saa.flatten(),
                    "p": p, "c": c, "s": s, "g": g,
                    "Q_saa": float(np.quantile(N_saa.flatten(), a)),
                    "Q_bay": float(np.quantile(N_bay.flatten(), a))}
    return out


# ════════════════════════════════════════════════════════════════════
# FIGURE 5 — posterior shrinkage: r and k Beta posteriors narrow with m
# (explains the crossover; ties to Hill 1997 — Bayesian carries parameter
#  uncertainty, wide at small m, collapsing toward the point estimate as m
#  grows, which is exactly why Bayesian differs from SAA only at small m)
# ════════════════════════════════════════════════════════════════════
def fig_posterior_shrinkage(out_path, category=None, period="W",
                            m_list=(5, 10, 20, 50), param="r"):
    """
    Beta posterior of r (or k) for increasing training sizes m, one curve
    per m, for a single category. Wider at small m (more uncertainty),
    narrowing as m grows. param in {"r","k"}.
    """
    from scipy.stats import beta as beta_dist
    from data_analysis import order_items_clean, returns_clean
    from preprocessing import build_category_datasets
    from bayesian_models import build_posteriors

    datasets = build_category_datasets(order_items_clean, returns_clean, period=period)
    if category is None:
        category = next(iter(datasets))
    panel = datasets[category]["panel"]
    m_max = len(panel)
    ms = [m for m in m_list if m <= m_max] + [m_max]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(ms)))
    xx = np.linspace(0, 1, 600)
    for m, col in zip(ms, colors):
        post = build_posteriors(panel.iloc[:m])
        a, b = post[param]["a_post"], post[param]["b_post"]
        pdf = beta_dist.pdf(xx, a, b)
        lbl = f"m = {m}" if m != m_max else f"m = all ({m_max})"
        ax.plot(xx, pdf, color=col, lw=2, label=lbl)
        ax.axvline(post[param]["post_mean"], color=col, ls=":", lw=1, alpha=0.6)
    # zoom x to the region where mass actually is
    full = build_posteriors(panel)
    mu = full[param]["post_mean"]
    ax.set_xlim(max(0, mu - 0.12), min(1, mu + 0.12))
    pname = {"r": "return probability  r", "k": "resalability  k"}[param]
    ax.set_xlabel(f"Posterior of {pname}")
    ax.set_ylabel("Posterior density")
    ax.set_title(f"Posterior shrinkage — {pname}\n"
                 f"({category}, {period}; representative)", fontsize=11)
    ax.legend(frameon=False, title="Training size")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out_path}")


# ════════════════════════════════════════════════════════════════════
# FIGURE 6 — demand posterior predictive captures overdispersion
# (NegBin-Beta posterior-predictive VMR vs the data VMR, per category;
#  a Poisson-Gamma alternative would collapse to VMR~1 — the rationale
#  for choosing NegBin in bayesian_models.py)
# ════════════════════════════════════════════════════════════════════
def fig_demand_overdispersion(out_path, period="W", J=20000, seed=0):
    """
    Per category: variance-to-mean ratio (VMR) of the NegBin-Beta posterior
    predictive vs the VMR of the observed gross demand. Points on the y=x
    line mean the model reproduces the observed spread; a Poisson model
    would sit near VMR=1 regardless of the data.
    """
    from data_analysis import order_items_clean, returns_clean
    from preprocessing import build_category_datasets
    from bayesian_models import build_posteriors, sample_demand_posterior

    datasets = build_category_datasets(order_items_clean, returns_clean, period=period)
    rng = np.random.default_rng(seed)
    cats, vmr_data, vmr_pp = [], [], []
    for cat, ds in datasets.items():
        G = ds["panel"]["G_t"].values.astype(float)
        post = build_posteriors(ds["panel"])
        dem = post["demand"]
        G_pp = sample_demand_posterior(dem["r_NB"], dem["a_post"], dem["b_post"], J, rng)
        cats.append(cat)
        vmr_data.append(G.var() / G.mean())
        vmr_pp.append(G_pp.var() / G_pp.mean())

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    lim = max(max(vmr_data), max(vmr_pp)) * 1.15
    ax.plot([0, lim], [0, lim], color="0.6", ls="--", lw=1.2,
            label="y = x  (perfect match)")
    ax.axhline(1.0, color="#E76F51", ls=":", lw=1.2,
               label="VMR = 1  (Poisson limit)")
    ax.scatter(vmr_data, vmr_pp, s=90, color=METHOD_COLOR["Bayesian"], zorder=5)
    for c, xd, yp in zip(cats, vmr_data, vmr_pp):
        ax.annotate(c, (xd, yp), fontsize=8, xytext=(5, 4),
                    textcoords="offset points")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("VMR of observed demand (data)")
    ax.set_ylabel("VMR of NegBin-Beta posterior predictive")
    ax.set_title("Demand model captures overdispersion", fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    OUT = Path("comparison_outputs"); OUT.mkdir(exist_ok=True)

    # Results figures (from CSVs)
    df = pd.read_csv(OUT / "method_comparison.csv"); df["m_train"] = df["m_train"].astype(str)
    fig_crossover(df, OUT / "fig_crossover.png")
    # cost and Q vs sample size (same layout, method comparison per period)
    fig_metric_vs_samplesize(
        df, OUT / "fig_cost_vs_samplesize.png",
        value_col="cost_mean", se_col=None, ylabel="Expected cost",
        suptitle="Expected cost by sample size and period")
    if "Q_mean" in df.columns:
        fig_metric_vs_samplesize(
            df, OUT / "fig_Q_vs_samplesize.png",
            value_col="Q_mean", se_col="Q_se", ylabel="Order quantity  $Q^*$",
            suptitle="Order quantity $Q^*$ by sample size and period")
    seq = pd.read_csv(OUT / "sequencing_importance.csv")
    fig_sequencing(seq, OUT / "fig_sequencing.png")
    if "method" in seq.columns:
        for m in seq.method.unique():
            fig_sequencing_levels(seq, OUT / f"fig_sequencing_levels_{m}.png", method=m)
    else:
        fig_sequencing_levels(seq, OUT / "fig_sequencing_levels.png")

    # Model-descriptive figures (full-data fits)
    fits = _build_fulldata_fits(period="W", rule="S4")
    cat0 = next(iter(fits))
    d0 = fits[cat0]
    plot_net_demand_distribution(d0["N_saa"], d0["N_bay"], d0["Q_saa"], d0["Q_bay"],
                                 title=f"{cat0} (W)",
                                 save_path=OUT / "fig_net_demand_dist.png")
    plot_cost_curves(fits, save_path=OUT / "fig_cost_curves.png")

    # Bayesian posterior figures (tie the crossover to parameter uncertainty)
    fig_posterior_shrinkage(OUT / "fig_posterior_shrinkage_r.png", param="r")
    fig_posterior_shrinkage(OUT / "fig_posterior_shrinkage_k.png", param="k")
    fig_demand_overdispersion(OUT / "fig_demand_overdispersion.png")
    print("\nDone. Figures in comparison_outputs/.")