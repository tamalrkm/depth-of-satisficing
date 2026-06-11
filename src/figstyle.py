"""Shared publication figure style. Import for its side effects:  import figstyle
Sets a clean, colourblind-safe, Nature-leaning matplotlib style and exports helpers.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colourblind-safe palette
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
# semantic colours for the four time controls (used across figures)
TC_COLORS = {"classical": "#0072B2", "rapid": "#009E73", "blitz": "#E69F00", "bullet": "#999999"}

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.titleweight": "bold",
    "axes.labelcolor": "#222222",
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#cccccc",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.4,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": "#444444",
    "ytick.color": "#444444",
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "lines.markersize": 5,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
})


def panel_label(ax, letter, dx=-0.12, dy=1.04):
    """Place a bold panel label (a, b, ...) at the top-left of an axes."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left")


def save(fig, path):
    """Save both a high-res PNG (for the draft PDF) and a vector PDF (for submission)."""
    fig.savefig(path, dpi=300)
    fig.savefig(path.rsplit(".", 1)[0] + ".pdf")
    return path
