"""
visualization.py — All model output plots (Sections 4-6 figures).

Collects the plotting functions from the Colab code into one module.
Each function saves a figure to disk (default: config.FIGURES_DIR).

KEY FIX vs Colab: plot_return_rate_sensitivity() previously HARD-CODED
empirical r_hat values (0.0844, 0.0934, ...) from the OLD DIRTY data
(800 returns). Those are wrong for the cleaned data (376 returns, r_hat
~0.045-0.055). The empirical_r dict is now a REQUIRED PARAMETER, passed
in from parameter_estimation so the figure always reflects current data.

────────────────────────────────────────────────────────────────────
Colab origin: plot_* functions, lines 646-975.
  Changes: hardcoded empirical_r removed → passed as argument;
           save paths default to config.FIGURES_DIR;
           verbose prints kept (these run rarely).
────────────────────────────────────────────────────────────────────
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import FIGURES_DIR


# ════════════════════════════════════════════════════════════════════
# 1. Net demand distribution (single category)
# ════════════════════════════════════════════════════════════════════
def plot_net_demand_distribution(N_matrix, Q_opt, Q_bench,
                                 title="", save_path=None):
    """
    Histogram of net demand N_t^(j) with stochastic and benchmark optima.
    Colab origin: lines 739-751.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "net_demand_dist.png"

    N_flat = N_matrix.flatten()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(N_flat, bins=50, density=True, alpha=0.65, color="steelblue",
            label="Net demand $N_t^{(j)}$")
    ax.axvline(Q_opt, color="blue", ls="--", lw=2,
               label=f"$Q^*$ stochastic = {Q_opt:.1f}")
    ax.axvline(Q_bench, color="red", ls="--", lw=2,
               label=f"$Q^*$ benchmark = {Q_bench:.1f}")
    ax.set_xlabel("Net Demand")
    ax.set_ylabel("Density")
    ax.set_title(f"Net Demand Distribution — {title}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════
# 2. Cost curves EP(Q) — all categories
# ════════════════════════════════════════════════════════════════════
def plot_all_cost_curves(all_cost_data: dict, save_path=None):
    """
    Multi-panel EP(Q) cost curves for all categories.

    all_cost_data: {cat: {N_flat, c, s, pN, gN, Q_stoch, Q_bench}}
    Colab origin: lines 646-736.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "cost_curves_all_categories.png"

    categories = list(all_cost_data.keys())
    n = len(categories)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 9))
    axes_flat = axes.flatten()

    for ax_idx, cat in enumerate(categories):
        ax = axes_flat[ax_idx]
        d = all_cost_data[cat]
        N_flat = d["N_flat"]
        cu = d["pN"] - d["c"] + d["gN"]
        co = d["c"] - d["s"]

        Q_grid = np.linspace(N_flat.min() * 0.3, N_flat.max() * 1.5, 400)
        EP = np.array([
            cu * np.mean(np.maximum(N_flat - q, 0))
            + co * np.mean(np.maximum(q - N_flat, 0))
            for q in Q_grid
        ])
        EP_stoch = float(cu * np.mean(np.maximum(N_flat - d["Q_stoch"], 0))
                         + co * np.mean(np.maximum(d["Q_stoch"] - N_flat, 0)))
        EP_bench = float(cu * np.mean(np.maximum(N_flat - d["Q_bench"], 0))
                         + co * np.mean(np.maximum(d["Q_bench"] - N_flat, 0)))

        ax.plot(Q_grid, EP, color="#2166ac", lw=2, label="EP(Q)")
        ax.axvline(d["Q_stoch"], color="#2166ac", ls="--", lw=1.8,
                   label=f"$Q^*_{{\\rm stoch}}={d['Q_stoch']:.0f}$")
        ax.scatter(d["Q_stoch"], EP_stoch, color="#2166ac", s=70, zorder=5)
        ax.axvline(d["Q_bench"], color="#d6604d", ls="--", lw=1.8,
                   label=f"$Q^*_{{\\rm bench}}={d['Q_bench']:.1f}$")
        ax.scatter(d["Q_bench"], EP_bench, color="#d6604d", s=70, zorder=5)
        ax.fill_between(
            [min(d["Q_stoch"], d["Q_bench"]), max(d["Q_stoch"], d["Q_bench"])],
            0, max(EP_stoch, EP_bench), alpha=0.12, color="#fdae61",
        )
        ax.axhline(EP_stoch, color="#2166ac", ls=":", lw=1.2, alpha=0.7)
        ax.axhline(EP_bench, color="#d6604d", ls=":", lw=1.2, alpha=0.7)
        ax.set_title(cat, fontsize=11, fontweight="bold", pad=5)
        ax.set_xlabel("Order Quantity $Q$", fontsize=9)
        ax.set_ylabel("Expected Total Cost $EP(Q)$", fontsize=9)
        ax.legend(fontsize=7.5, loc="upper right")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=8)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle(
        "SAA Expected Cost Curve $EP(Q)$ — All Categories\n"
        "Dashed: stochastic optimum (blue), deterministic netting "
        "benchmark (red). Dotted: corresponding minimum costs.",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════
# 3. Sensitivity lines (single category): Q* vs k, one line per g
# ════════════════════════════════════════════════════════════════════
def plot_sensitivity_lines(sens_df, title="", save_path=None):
    """Q* vs k_hat, one line per g. Colab origin: lines 754-776."""
    if save_path is None:
        save_path = FIGURES_DIR / "sensitivity_lines.png"

    sens_df = sens_df.sort_values(["g", "k_hat"])
    g_vals = sorted(sens_df["g"].unique())

    fig, ax = plt.subplots(figsize=(9, 5))
    for g_val in g_vals:
        sub = sens_df[sens_df["g"] == g_val]
        ax.plot(sub["k_hat"], sub["Q_opt"], marker="o", lw=2, label=f"g={g_val:.1f}")
    ax.set_xlabel("Resalability Probability $k$")
    ax.set_ylabel("Optimal Order Quantity $Q^*$")
    ax.set_title(f"Sensitivity of $Q^*$ to $k$ and $g$ — {title}")
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════
# 4. Return-rate sensitivity (all categories, two panels)
# ════════════════════════════════════════════════════════════════════
def plot_return_rate_sensitivity(all_r_sens: dict,
                                 empirical_r: dict,
                                 save_path=None):
    """
    Two-panel Q*(r_hat) and EP(Q*)(r_hat) for all categories.

    CHANGED vs Colab: empirical_r is now a REQUIRED ARGUMENT (was hardcoded
    with dirty-data values). Pass {cat: r_hat} computed from current data.

    Colab origin: lines 778-869.
    """
    if save_path is None:
        save_path = FIGURES_DIR / "sensitivity_return_rate.png"

    palette = {
        "Classic Scrunchies":     "#1f77b4",
        "Luxury Silk Scrunchies": "#d62728",
        "Seasonal Prints":        "#2ca02c",
        "Mini Scrunchies":        "#ff7f0e",
        "Scrunchie Bundles":      "#9467bd",
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for cat, df_r in all_r_sens.items():
        color = palette.get(cat, "grey")
        r_emp = empirical_r.get(cat, None)

        ax1.plot(df_r["r_hat"], df_r["Q_opt"], color=color, lw=2,
                 marker="o", ms=4, label=cat)
        if r_emp is not None:
            q_emp = df_r.loc[(df_r["r_hat"] - r_emp).abs().idxmin(), "Q_opt"]
            ax1.scatter(r_emp, q_emp, color=color, s=100, zorder=5,
                        edgecolors="black", linewidths=0.8)

        ax2.plot(df_r["r_hat"], df_r["cost"], color=color, lw=2,
                 marker="o", ms=4, label=cat)
        if r_emp is not None:
            c_emp = df_r.loc[(df_r["r_hat"] - r_emp).abs().idxmin(), "cost"]
            ax2.scatter(r_emp, c_emp, color=color, s=100, zorder=5,
                        edgecolors="black", linewidths=0.8)

    if empirical_r:
        r_min_obs = min(empirical_r.values())
        r_max_obs = max(empirical_r.values())
        for ax in (ax1, ax2):
            ax.axvspan(r_min_obs, r_max_obs, alpha=0.10, color="grey",
                       label="Observed $\\hat{r}$ range" if ax is ax1 else None)

    for ax in (ax1, ax2):
        ax.set_xlabel(r"Return Rate $\hat{r}$", fontsize=11)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=9)

    ax1.set_ylabel(r"Optimal Order Quantity $Q^*$", fontsize=11)
    ax1.set_title(r"$Q^*$ vs Return Rate $\hat{r}$", fontsize=12, fontweight="bold")
    ax2.set_ylabel(r"Expected Total Cost $EP(Q^*)$", fontsize=11)
    ax2.set_title(r"$EP(Q^*)$ vs Return Rate $\hat{r}$", fontsize=12, fontweight="bold")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.13))
    fig.suptitle(
        r"Sensitivity of $Q^*$ and $EP(Q^*)$ to Return Rate $\hat{r}$"
        "\n(all other parameters fixed at category baseline)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════
# 5. Summary bar chart (all categories): Q* stochastic vs benchmark
# ════════════════════════════════════════════════════════════════════
def plot_all_categories_summary(summary: pd.DataFrame, save_path=None):
    """Bar chart: Q* stochastic vs benchmark per category. Colab: 870-890."""
    if save_path is None:
        save_path = FIGURES_DIR / "summary_all_categories.png"

    x = np.arange(len(summary))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, summary["Q_opt_stochastic"], width,
           label="$Q^*$ Stochastic", color="#4C72B0")
    ax.bar(x + width/2, summary["Q_opt_benchmark"], width,
           label="$Q^*$ Benchmark", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["category"], rotation=15, ha="right")
    ax.set_ylabel("Optimal Order Quantity $Q^*$")
    ax.set_title("Optimal $Q^*$ by Category — Stochastic vs Benchmark")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    for i, row in summary.iterrows():
        shift = row["Q_opt_stochastic"] - row["Q_opt_benchmark"]
        ax.text(i, max(row["Q_opt_stochastic"], row["Q_opt_benchmark"]) + 0.3,
                f"{shift:+.1f}", ha="center", fontsize=9, color="grey")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════
# 6. Sensitivity panels (all categories): Q* vs k, one panel per category
# ════════════════════════════════════════════════════════════════════
def plot_all_sensitivity_panels(all_sens: dict, save_path=None):
    """Multi-panel Q* vs k_hat (one panel per category). Colab: 892-928."""
    if save_path is None:
        save_path = FIGURES_DIR / "sensitivity_all_categories.png"

    categories = list(all_sens.keys())
    n = len(categories)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows),
                             sharey=False, sharex=True)
    axes_flat = axes.flatten()

    for ax, cat_name in zip(axes_flat, categories):
        sens_df = all_sens[cat_name].sort_values(["g", "k_hat"])
        g_vals = sorted(sens_df["g"].unique())
        for g_val in g_vals:
            sub = sens_df[sens_df["g"] == g_val]
            ax.plot(sub["k_hat"], sub["Q_opt"], marker="o", lw=2,
                    label=f"g={g_val:.1f}")
        ax.set_title(cat_name, fontsize=11)
        ax.set_xlabel("Resalability Probability $\\hat{k}$")
        ax.set_ylabel("Optimal Order Quantity $Q^*$")
        ax.legend(ncol=2, fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Sensitivity of $Q^*$ to Resalability $\\hat{k}$ and "
                 "Shortage Penalty $g$ — All Categories",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ════════════════════════════════════════════════════════════════════
# Standalone test (run: python visualization.py)
# ════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from data_analysis import order_items_clean, returns_clean
    from preprocessing import build_category_datasets
    from parameter_estimation import build_category_params
    from monte_carlo import build_net_demand_matrix
    from saa_solver import solve_saa
    from benchmark import netting_benchmark
    from gurobi_env import make_gurobi_env

    datasets = build_category_datasets(order_items_clean, returns_clean, period="W")
    params = build_category_params(datasets, order_items_clean)
    env = make_gurobi_env()

    # Build one figure to confirm the module works: net demand dist for Classic
    cat = "Classic Scrunchies"
    ds = datasets[cat]
    prm = params[cat]
    N = build_net_demand_matrix(ds, prm["r_hat"], prm["k_hat"], rule="S4", J=200, seed=0)
    saa = solve_saa(N, prm["c"], prm["s"], prm["pN"], prm["gN"], gurobi_env=env)
    bench = netting_benchmark(ds["panel"], prm["r_hat"], prm["k_hat"],
                              prm["c"], prm["s"], prm["pN"], prm["gN"], gurobi_env=env)

    plot_net_demand_distribution(
        N, saa["Q_opt"], bench["Q_opt"],
        title=f"{cat} (W, S4)",
    )
    print("\nVisualization module OK — figure saved to results/figures/")