#!/usr/bin/env python
"""
Slide table: CLMBR linear probe (stock FEMR) vs. sklearn LogisticRegression (9e).

Highlights the two rows that differ substantively; everything else is either
identical or equivalent at convergence.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

NAVY, STEEL, INK, MUTED = "#1B3A57", "#4E80B8", "#222222", "#6B7680"
ROW_A, ROW_B, HILITE, RULE = "#FFFFFF", "#F5F7F9", "#EAF1F8", "#DDE3E8"

COLS = ("CLMBR linear probe\n(stock FEMR)", "sklearn LogisticRegression\n(our 9e)")

# (label, stock, ours, differs_substantively)
ROWS = [
    ("Model",           "L2-regularised logistic regression",
                        "L2-regularised logistic regression",            False),
    ("Objective",       "mean(BCE) + ½·l2·||β||²",
                        "½·||w||² + C·Σ BCE",                            False),
    ("Optimiser",       "conjugate gradient (JAX, hand-written)",
                        "lbfgs (quasi-Newton)",                          False),
    ("Penalty scale",   "l2",
                        "C = 1 / (l2 · n_train)",                        True),
    ("L2 grid",         "20 points, 10¹ → 10⁻⁵, plus 0",
                        "identical",                                     False),
    ("Intercept",       "appended ones column, unpenalised",
                        "fit_intercept=True, unpenalised",               False),
    ("Model selection", "best dev AUROC",
                        "best valid AUROC",                              False),
    ("Split source",    "CLMBR hash 80/5/15, seed 97",
                        "cohort file, patient-level",                    True),
    ("Deterministic",   "yes",                "yes",                     False),
    ("Role here",       "reference implementation",
                        "comparable-to-baselines run",                   False),
]

XL, XM, XR, XE = 0.035, 0.315, 0.655, 0.985      # column edges
RH, HH = 0.0715, 0.115                            # row / header heights
TOP = 0.905

fig, ax = plt.subplots(figsize=(12.4, 6.6), dpi=200)
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

# header
hy = TOP - HH
ax.add_patch(Rectangle((XL, hy), XE - XL, HH, facecolor=NAVY, edgecolor="none"))
for x, txt in ((XM, COLS[0]), (XR, COLS[1])):
    ax.text(x + 0.014, hy + HH / 2, txt, fontsize=11.5, color="white",
            fontweight="medium", va="center", ha="left", linespacing=1.45)

# rows
y = hy
for i, (label, a, b, diff) in enumerate(ROWS):
    y -= RH
    bg = HILITE if diff else (ROW_A if i % 2 == 0 else ROW_B)
    ax.add_patch(Rectangle((XL, y), XE - XL, RH, facecolor=bg, edgecolor="none"))
    ax.plot([XL, XE], [y, y], color=RULE, lw=0.7, zorder=2)

    ax.text(XL + 0.014, y + RH / 2, label, fontsize=10.5,
            color=NAVY if diff else INK,
            fontweight="medium" if diff else "normal", va="center", ha="left")
    for x, txt in ((XM, a), (XR, b)):
        ax.text(x + 0.014, y + RH / 2, txt, fontsize=10.2,
                color=NAVY if diff else INK,
                fontweight="medium" if diff else "normal",
                va="center", ha="left")
    if diff:
        ax.add_patch(Rectangle((XL, y), 0.0045, RH, facecolor=STEEL, edgecolor="none"))

# column separators
for x in (XM, XR):
    ax.plot([x, x], [y, TOP], color=RULE, lw=0.8, zorder=1)
ax.plot([XL, XE], [y, y], color=NAVY, lw=1.4, zorder=3)

ax.text(XL, TOP + 0.038,
        "Same estimator, same convex objective — only two differences are substantive",
        fontsize=13.5, fontweight="bold", color=NAVY, ha="left", va="bottom")
ax.text(XL, y - 0.045,
        "Highlighted rows differ substantively. The objective row is equivalent once "
        "C = 1/(l2·n_train): dividing sklearn's objective by C·n recovers FEMR's exactly.\n"
        "Both optimisers minimise a strictly convex function with a unique global optimum, "
        "so they converge to the same coefficients.",
        fontsize=9.6, color=MUTED, ha="left", va="top", linespacing=1.6)

plt.tight_layout()
for ext in ("png", "svg"):
    fig.savefig(f"probe_comparison_table.{ext}", bbox_inches="tight", facecolor="white")
print("wrote probe_comparison_table.png / .svg")
