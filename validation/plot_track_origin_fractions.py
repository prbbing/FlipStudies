"""
Standalone script: track origin composition by jet flavour.

Extracted from plot_track_d0_by_origin.py — computes the fraction of valid
tracks belonging to each GN2v01_trackOrigin class, separately for b-, c-,
and light-jets, and plots:
  1. a stacked bar chart (composition per flavour), and
  2. a grouped bar chart (per-origin fraction across flavours).
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

H5_FILE  = "mc-flavtag-ttbar-small.h5"
N_JETS   = 200_000
D0_CUT   = 3.5        # |d0| < D0_CUT to flag valid tracks
Z0_CUT   = 5.0        # |z0SinTheta| < Z0_CUT to flag valid tracks
PLOT_DIR = "./track_origin_plots/"

FLAVOUR_TO_LABEL = {5: 0, 4: 1, 0: 2}
CLASS_NAMES      = ["b-jet", "c-jet", "light-jet"]
FLAVOUR_COLOURS  = ["#1f77b4", "#ff7f0e", "#2ca02c"]

ORIGIN_NAMES = [
    "Pileup",      # 0
    "Fake",        # 1
    "Primary",     # 2
    "From b",      # 3
    "From b→c",    # 4
    "From c",      # 5
    "From τ",      # 6
    "Other sec.",  # 7
]
N_ORIGINS = 8

os.makedirs(PLOT_DIR, exist_ok=True)

# ── load data ─────────────────────────────────────────────────────────
print(f"Loading {N_JETS:,} jets from {H5_FILE}...")
with h5py.File(H5_FILE, "r") as f:
    all_flavour = f["jets"]["HadronConeExclTruthLabelID"][:N_JETS]
    keep_jet    = np.isin(all_flavour, list(FLAVOUR_TO_LABEL.keys()))
    jet_idx     = np.where(keep_jet)[0]

    flavour_label = np.array([FLAVOUR_TO_LABEL[v] for v in all_flavour[keep_jet]])

    valid       = f["tracks"]["valid"][jet_idx]                          # (N, K) bool
    d0          = f["tracks"]["d0"][jet_idx].astype(np.float32)
    z0sintheta  = f["tracks"]["z0SinTheta"][jet_idx].astype(np.float32)
    origin      = f["tracks"]["GN2v01_trackOrigin"][jet_idx].astype(np.int8)

N, K = valid.shape
print(f"Jets kept: {N:,}  |  tracks per jet (K): {K}")

# Valid track mask: flagged valid + |d0| < D0_CUT + |z0SinTheta| < Z0_CUT
track_valid = valid & (np.abs(d0) < D0_CUT) & (np.abs(z0sintheta) < Z0_CUT)   # (N, K)

# Flatten everything to 1-D arrays
flat_valid  = track_valid.ravel()                     # (N*K,)
flat_origin = origin.ravel()                          # (N*K,)
flat_flav   = np.repeat(flavour_label, K)             # (N*K,)


# ── origin fraction plots ─────────────────────────────────────────────
# Compute fraction of valid tracks from each origin, per jet flavour.
origin_counts = np.zeros((3, N_ORIGINS), dtype=np.int64)
for flav_idx in range(3):
    mask = flat_valid & (flat_flav == flav_idx)
    for orig_idx in range(N_ORIGINS):
        origin_counts[flav_idx, orig_idx] = (mask & (flat_origin == orig_idx)).sum()

totals    = origin_counts.sum(axis=1, keepdims=True)          # (3, 1)
fractions = origin_counts / np.where(totals > 0, totals, 1)   # (3, 8)

cmap_orig    = plt.get_cmap("tab10")
origin_cols  = [cmap_orig(i) for i in range(N_ORIGINS)]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("Track origin composition by jet flavour", fontweight="bold", fontsize=13)

# ── left: stacked bar per flavour ─────────────────────────────────────
ax = axes[0]
x  = np.arange(3)
bottoms = np.zeros(3)
for orig_idx in range(N_ORIGINS):
    frac = fractions[:, orig_idx]
    bars = ax.bar(x, frac, bottom=bottoms, color=origin_cols[orig_idx],
                  label=f"{orig_idx}: {ORIGIN_NAMES[orig_idx]}", edgecolor="white", linewidth=0.4)
    # annotate segments that are large enough to read
    for xi, (bot, f) in enumerate(zip(bottoms, frac)):
        if f > 0.03:
            ax.text(xi, bot + f / 2, f"{f:.1%}", ha="center", va="center",
                    fontsize=7, color="white", fontweight="bold")
    bottoms += frac

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, fontsize=10)
ax.set_ylabel("Fraction of valid tracks")
ax.set_ylim(0, 1)
ax.set_title("Stacked composition per flavour")
ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 1.0))

# ── right: grouped bars — one group per origin, bars = flavours ───────
ax   = axes[1]
w    = 0.22
offsets = np.array([-w, 0, w])
ox   = np.arange(N_ORIGINS)
for flav_idx, (flav_name, colour) in enumerate(zip(CLASS_NAMES, FLAVOUR_COLOURS)):
    ax.bar(ox + offsets[flav_idx], fractions[flav_idx], width=w,
           color=colour, label=flav_name, edgecolor="white", linewidth=0.4)

ax.set_xticks(ox)
ax.set_xticklabels(
    [f"{i}\n{ORIGIN_NAMES[i]}" for i in range(N_ORIGINS)],
    fontsize=7,
)
ax.set_ylabel("Fraction of valid tracks")
ax.set_title("Per-origin fraction by flavour")
ax.legend(fontsize=8)
ax.set_ylim(0, fractions.max() * 1.15)

plt.tight_layout()
out = os.path.join(PLOT_DIR, "track_origin_fractions.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {out}")

print("Done.")
