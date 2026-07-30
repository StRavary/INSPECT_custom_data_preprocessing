#!/usr/bin/env python
"""
Paired delta plot: MOTOR linear-probe test AUROC, CLMBR hash split minus
cohort (canonical) split, one point per INSPECT task.

Tasks are ordered by cohort-split AUROC (hardest first) to show that the
magnitude of disagreement tracks task difficulty -- the signature of sampling
noise rather than a systematic effect of the split mechanism.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# INSPECT's published task order, kept so this figure lines up with the
# grouped-bar chart of the same results.
SHORT  = ["PE", "1-mo\nMortality", "6-mo\nMortality", "12-mo\nMortality",
          "1-mo\nReadmission", "6-mo\nReadmission", "12-mo\nReadmission",
          "12-mo\nPH"]
COHORT = np.array([0.682, 0.936, 0.910, 0.897, 0.783, 0.770, 0.757, 0.851])
HASH   = np.array([0.705, 0.930, 0.907, 0.899, 0.759, 0.792, 0.779, 0.853])

NAVY, STEEL, GREY, MUTED = "#1B3A57", "#4E80B8", "#9AA5AD", "#6B7680"

d = HASH - COHORT
labels = [f"{t}\n({a:.3f})" for t, a in zip(SHORT, COHORT)]

mean = d.mean()
se = d.std(ddof=1) / np.sqrt(len(d))
tcrit = 2.364624                                  # t(7), two-sided 95%
lo, hi = mean - tcrit * se, mean + tcrit * se
tstat = mean / se


def t_sf(t, df):
    x = np.linspace(abs(t), abs(t) + 60, 400001)
    c = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2))
    return float(np.trapezoid(c * (1 + x ** 2 / df) ** (-(df + 1) / 2), x))


pval = 2 * t_sf(tstat, len(d) - 1)

fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=200)

ax.axhspan(lo, hi, color=GREY, alpha=0.22, zorder=0)
ax.axhline(mean, color=GREY, lw=1.6, ls="--", zorder=1)
ax.axhline(0, color="#333333", lw=1.4, zorder=2)

x = np.arange(len(d))
colors = [STEEL if v > 0 else NAVY for v in d]
ax.vlines(x, 0, d, color=colors, lw=3.2, zorder=3)
ax.scatter(x, d, s=100, c=colors, zorder=4, edgecolor="white", linewidth=1.3)

for xi, v in zip(x, d):
    ax.text(xi, v + (0.0022 if v > 0 else -0.0022), f"{v:+.3f}",
            ha="center", va="bottom" if v > 0 else "top",
            fontsize=10.5, color="#222222", zorder=5)

# band label, placed in empty space inside the CI band
ax.text(len(d) - 0.55, hi - 0.0015, "95% CI of mean", ha="right", va="top",
        fontsize=9, color=MUTED, style="italic")

ax.annotate("favours hash split", xy=(-0.42, 0.0305), fontsize=9.5,
            color=STEEL, ha="left", va="center")
ax.annotate("favours cohort split", xy=(-0.42, -0.0305), fontsize=9.5,
            color=NAVY, ha="left", va="center")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.tick_params(axis="x", length=0, pad=10)
ax.set_xlim(-0.6, len(d) - 0.4)
ax.set_ylim(-0.035, 0.035)
ax.set_yticks(np.arange(-0.03, 0.031, 0.01))
ax.set_ylabel("Δ test AUROC   (hash split − cohort split)", fontsize=11.5)

ax.set_title("Split source has no detectable effect on MOTOR performance",
             fontsize=15, fontweight="bold", color=NAVY, pad=30)
ax.text(0.5, 1.045,
        f"mean {mean:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}]   ·   paired t = {tstat:.2f}, "
        f"p = {pval:.2f}   ·   {(d > 0).sum()}/8 favour hash",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=10.5, color=MUTED)
ax.text(0.5, -0.205,
        "cohort-split AUROC in parentheses — the largest differences fall on the "
        "lowest-AUROC tasks, consistent with sampling noise",
        transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=MUTED)

for s in ("top", "right", "bottom"):
    ax.spines[s].set_visible(False)
ax.spines["left"].set_color("#CCCCCC")
ax.grid(axis="y", color="#EDEDED", lw=0.8, zorder=0)
ax.set_axisbelow(True)

plt.subplots_adjust(bottom=0.27, top=0.84)
for ext in ("png", "svg"):
    fig.savefig(f"motor_split_delta.{ext}", bbox_inches="tight", facecolor="white")
print(f"mean {mean:+.4f}  SE {se:.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  t={tstat:.2f}  p={pval:.3f}")
print("wrote motor_split_delta.png / .svg")
