"""Played-move regret vs engine depth, by rating band, for all / swing-up / swing-down decisions.
Regime selectable: pooled (all TCs + OTB), slow (classical+rapid), otb (broadcast), or a single TC.
    uv run python src/fig_swing_compare.py [all|slow|otb|classical|rapid]   (default: slow)
"""
import sys, numpy as np, torch, yaml
import figstyle
import matplotlib.pyplot as plt
from analyze import band_of, BANDS, BAND_MID

cfg = yaml.safe_load(open("config.yaml"))
grid = np.array(cfg["model"]["depth_grid"])
b = torch.load(cfg["data"]["train_tensor"]); meta = b["meta"]
y = b["y"]; pr = b["delta"][np.arange(len(y)), y].numpy()
elo = np.array(meta["elo"]); sw = np.array(meta["swing"])
tc = np.array(meta["time_class"]); src = np.array(meta["source"])
bands = np.array([band_of(e) for e in elo])

print("composition of the 922k primary set:")
print("  time_class:", {t: int((tc == t).sum()) for t in sorted(set(tc))})
print("  source:", {s: int((src == s).sum()) for s in set(src)})

regime = sys.argv[1] if len(sys.argv) > 1 else "slow"
if regime == "all":      rmask, rname = np.ones(len(y), bool), "all TCs + OTB pooled"
elif regime == "slow":   rmask, rname = np.isin(tc, ["classical", "rapid"]), "slow controls (classical+rapid)"
elif regime == "otb":    rmask, rname = src == "broadcast", "OTB broadcast only"
else:                    rmask, rname = tc == regime, f"{regime} only"
print(f"\nregime = {rname}: n={int(rmask.sum()):,}")

panels = [("all decisions", np.ones(len(y), bool)),
          ("swing-up (deep-discovery)", sw == "up"),
          ("swing-down (traps)", sw == "down")]
fig, ax = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
cmap = plt.cm.viridis(np.linspace(0, 1, len(BANDS)))
for (title, m0), axi, lab in zip(panels, ax, "abc"):
    m = m0 & rmask
    for bi in range(len(BANDS)):
        sel = (bands == bi) & m
        if sel.sum() > 50:
            axi.plot(grid, pr[sel].mean(0), color=cmap[bi], label=f"{BAND_MID[bi]}")
    axi.set_title(f"{title}  (n={int(m.sum()):,})"); axi.set_xlabel("engine search depth")
    axi.grid(alpha=.3); figstyle.panel_label(axi, lab)
    bm = np.array([pr[(bands == bi) & m, -1].mean() if ((bands == bi) & m).sum() > 50 else np.nan
                   for bi in range(len(BANDS))])
    print(f"  {title:28s} deep-regret by band: " + " ".join(f"{v:.3f}" for v in bm if not np.isnan(v))
          + f"   spread={np.nanmax(bm)-np.nanmin(bm):.3f}")
ax[0].set_ylabel("mean regret of played move (win-prob)")
ax[2].legend(title="rating", fontsize=7, ncol=2)
fig.suptitle(rname, y=1.02, fontsize=11)
fig.tight_layout()
p = f"paper/figs_irt/fig1_swing_compare_{regime}.png"; fig.savefig(p, dpi=110, bbox_inches="tight")
print("->", p)
