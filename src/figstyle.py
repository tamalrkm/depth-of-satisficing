"""Shared publication figure style. Import for its side effects:  import figstyle
Print-first, colourblind-safe style for the manuscript figures.

Colour is assigned by job, not by taste (palette validated with a Machado-2009
CVD separation check):
  * categorical (time controls)  -- blue / aqua / amber, fixed order; bullet is
    deliberately a de-emphasis grey (it is the negative control, not a series
    competing for attention). Aqua and amber sit below 3:1 on white, so every
    figure that uses them also direct-labels the lines (the relief rule).
  * sequential (rating bands)    -- one blue ramp, light->dark = weak->strong,
    monotone lightness; extremes direct-labelled in the figures.
  * status (move severity)       -- good/warning/serious/critical, reserved for
    best/inaccuracy/mistake/blunder only.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- palettes
# text tokens: marks wear colour, text wears ink.
INK      = "#0b0b0b"   # primary text
INK2     = "#52514e"   # secondary text (axis labels, ticks)
MUTED    = "#8a8987"   # muted text / reference lines
GRID     = "#e8e7e4"   # hairline grid, one step off the white surface

# categorical: time controls (fixed order, never cycled)
TC_COLORS = {"classical": "#2a78d6", "rapid": "#1baf7a",
             "blitz": "#eda100", "bullet": MUTED}
TC_ORDER = ["classical", "rapid", "blitz", "bullet"]

# sequential: 8 rating bands, one blue hue, light->dark (weak->strong)
BAND_RAMP = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
             "#2a78d6", "#256abf", "#184f95", "#0d366b"]

# ordinal: 3 swing terciles (same blue hue, big lightness steps)
TERCILE_RAMP = ["#86b6ef", "#2a78d6", "#0d366b"]

# status: move severity (reserved semantics, never reused as "series 4")
SEVERITY = {"best": "#0ca30c", "inaccuracy": "#fab219",
            "mistake": "#ec835a", "blunder": "#d03b3b"}

# single-measure emphasis pair (one hue: emphasised vs de-emphasised)
ACCENT       = "#2a78d6"   # blue-450
ACCENT_DARK  = "#184f95"   # blue-600
ACCENT_LIGHT = "#9ec5f4"   # blue-200

PALETTE = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#e87ba4", "#eb6834", INK]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.titleweight": "regular",
    "axes.titlelocation": "left",
    "axes.titlecolor": INK,
    "axes.titlepad": 8,
    "axes.labelsize": 9,
    "axes.labelcolor": INK2,
    "axes.edgecolor": INK2,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,          # grid under the data, always
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.alpha": 1.0,               # hairline solid, recessive by colour not alpha
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
    "lines.markersize": 5.5,
    "axes.prop_cycle": plt.cycler(color=PALETTE),
})


def panel_label(ax, letter, dx=-0.12, dy=1.06):
    """Bold panel letter (a, b, ...) at the top-left, Nature-style."""
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=12, fontweight="bold",
            color=INK, va="top", ha="left")


def direct_label(ax, x, y, text, color, dx=4, dy=0, fontsize=8, ha="left", va="center",
                 weight="regular"):
    """Direct series label at a line end (identity never rides on colour alone)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=fontsize, color=color, ha=ha, va=va, fontweight=weight,
                annotation_clip=False)


def ring(**kw):
    """Marker kwargs for the 'surface ring': white halo so dots stay legible on lines."""
    d = dict(markeredgecolor="white", markeredgewidth=1.1)
    d.update(kw)
    return d


def zero_line(ax, axis="y"):
    """Recessive zero/reference line."""
    if axis == "y":
        ax.axhline(0, color=MUTED, lw=0.9, zorder=1)
    else:
        ax.axvline(0, color=MUTED, lw=0.9, zorder=1)


def save(fig, path):
    """Save both a high-res PNG (for the draft PDF) and a vector PDF (for submission)."""
    fig.savefig(path, dpi=300)
    fig.savefig(path.rsplit(".", 1)[0] + ".pdf")
    return path
