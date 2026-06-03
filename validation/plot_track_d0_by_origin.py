"""
Plot d0/z0 track impact-parameter variables categorised by
jet-flavour × GN2v01_trackOrigin combinations.

For each variable a 2×4 grid is produced (one subplot per origin class),
with three overlaid histograms coloured by jet flavour.
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

# Variables: (hdf5_field, axis_label, symmetric_clip_percentile)
VARIABLES = [
    ("d0",                                    r"$d_0$ [mm]",                99),
    ("z0SinTheta",                            r"$z_0\sin\theta$ [mm]",      99),
    ("lifetimeSignedD0",                      r"Signed $d_0$ [mm]",         99),
    ("lifetimeSignedZ0SinTheta",              r"Signed $z_0\sin\theta$ [mm]", 99),
    ("lifetimeSignedD0Significance",          r"$d_0$ significance",        99),
    ("lifetimeSignedZ0SinThetaSignificance",  r"$z_0\sin\theta$ significance", 99),
]

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

    var_arrays = {}
    for field, _, _ in VARIABLES:
        var_arrays[field] = f["tracks"][field][jet_idx].astype(np.float32)

N, K = valid.shape
print(f"Jets kept: {N:,}  |  tracks per jet (K): {K}")

# Valid track mask: flagged valid + |d0| < D0_CUT + |z0SinTheta| < Z0_CUT
track_valid = valid & (np.abs(d0) < D0_CUT) & (np.abs(z0sintheta) < Z0_CUT)   # (N, K)

# Flatten everything to 1-D arrays
flat_valid  = track_valid.ravel()                     # (N*K,)
flat_origin = origin.ravel()                          # (N*K,)
flat_flav   = np.repeat(flavour_label, K)             # (N*K,)


# ── plotting helper ───────────────────────────────────────────────────
def plot_variable(field, xlabel, clip_pct):
    flat_vals = var_arrays[field].ravel()

    fig, axes = plt.subplots(2, 4, figsize=(22, 9), sharey=False)
    fig.suptitle(
        f"{xlabel}  —  by track origin and jet flavour",
        fontweight="bold", fontsize=13,
    )
    axes = axes.ravel()

    for orig_idx in range(N_ORIGINS):
        ax        = axes[orig_idx]
        orig_mask = flat_valid & (flat_origin == orig_idx)
        n_total   = orig_mask.sum()

        # Determine symmetric x-range from all valid tracks of this origin
        if n_total > 0:
            all_vals = flat_vals[orig_mask]
            edge = np.percentile(np.abs(all_vals), clip_pct)
            xlim = (-edge, edge)
        else:
            xlim = (-1, 1)

        for flav_idx, (flav_name, colour) in enumerate(zip(CLASS_NAMES, FLAVOUR_COLOURS)):
            mask = orig_mask & (flat_flav == flav_idx)
            vals = flat_vals[mask]
            if vals.size == 0:
                continue
            ax.hist(
                vals, bins=60, range=xlim,
                histtype="step", density=False,
                label=f"{flav_name} ({mask.sum():,})",
                color=colour, linewidth=1.5,
            )

        ax.set_title(f"{orig_idx}: {ORIGIN_NAMES[orig_idx]}", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Counts", fontsize=8)
        ax.tick_params(labelsize=7)
        if orig_idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    safe_name = field.replace("/", "_")
    out = os.path.join(PLOT_DIR, f"track_{safe_name}_by_origin.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ── per-variable figures ──────────────────────────────────────────────
for field, xlabel, clip_pct in VARIABLES:
    plot_variable(field, xlabel, clip_pct)


# ── combined summary: all origins overlaid, one panel per flavour ─────
# Useful for seeing how the origin mix shifts between b/c/light jets.
for field, xlabel, clip_pct in VARIABLES[:4]:   # d0 and z0 only
    flat_vals = var_arrays[field].ravel()

    # Global clip
    all_valid_vals = flat_vals[flat_valid]
    edge = np.percentile(np.abs(all_valid_vals), clip_pct)
    xlim = (-edge, edge)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.suptitle(
        f"{xlabel}  —  all origins overlaid, by jet flavour",
        fontweight="bold", fontsize=13,
    )

    cmap   = plt.get_cmap("tab10")
    origin_colours = [cmap(i) for i in range(N_ORIGINS)]

    for flav_idx, (ax, flav_name) in enumerate(zip(axes, CLASS_NAMES)):
        flav_mask = flat_valid & (flat_flav == flav_idx)
        for orig_idx in range(N_ORIGINS):
            mask = flav_mask & (flat_origin == orig_idx)
            vals = flat_vals[mask]
            if vals.size == 0:
                continue
            ax.hist(
                vals, bins=60, range=xlim,
                histtype="step", density=False,
                label=f"{orig_idx}: {ORIGIN_NAMES[orig_idx]} ({mask.sum():,})",
                color=origin_colours[orig_idx], linewidth=1.5,
            )
        ax.set_title(flav_name, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Counts", fontsize=9)
        ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    safe_name = field.replace("/", "_")
    out = os.path.join(PLOT_DIR, f"track_{safe_name}_origins_per_flavour.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

# ── symmetry plots ───────────────────────────────────────────────────
# Two complementary views of symmetry around 0 for each variable:
#   1. Mirror overlay : f(x) solid vs f(−x) dashed on x ≥ 0 axis.
#                       Perfect overlap = symmetric.
#   2. Asymmetry      : A(x) = (f(x)−f(−x)) / (f(x)+f(−x)) for x > 0.
#                       Zero everywhere = symmetric.
# Both figures use a 2×4 grid of origin subplots, coloured by jet flavour.

N_BINS_SYM = 50


def _pos_neg_densities(vals, bins):
    """Return per-bin densities for x≥0 and |x| where x<0, normalised jointly."""
    n_total  = vals.size
    if n_total == 0:
        z = np.zeros(len(bins) - 1)
        return z, z
    bw = bins[1] - bins[0]
    pos, _ = np.histogram( vals[vals >= 0],  bins=bins)
    neg, _ = np.histogram(-vals[vals <  0],  bins=bins)
    # normalise by total count so both curves share the same scale
    return pos / (n_total * bw), neg / (n_total * bw)


def plot_symmetry_mirror(field, xlabel, clip_pct):
    """
    One cell per (flavour × origin) combination — 3 rows × 8 cols = 24 cells.
    Each cell: top panel f(x) vs f(−x), bottom panel ratio f(x)/f(−x).
    """
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    flat_vals = var_arrays[field].ravel()

    # Precompute per-origin symmetric x-range using all tracks (across flavours)
    origin_edges = {}
    for orig_idx in range(N_ORIGINS):
        vals = flat_vals[flat_valid & (flat_origin == orig_idx)]
        origin_edges[orig_idx] = np.percentile(np.abs(vals), clip_pct) if vals.size else 1.0

    fig = plt.figure(figsize=(8 * 3.2, 3 * 4.5))
    fig.suptitle(
        f"{xlabel}  —  mirror symmetry  "
        r"(solid $f(x)$,  dashed $f(-x)$ mirrored,  ratio $f(x)/f(-x)$)",
        fontweight="bold", fontsize=13, y=1.01,
    )

    outer = GridSpec(3, N_ORIGINS, figure=fig, hspace=0.55, wspace=0.35)

    for flav_idx, flav_name in enumerate(CLASS_NAMES):
        for orig_idx in range(N_ORIGINS):
            inner = GridSpecFromSubplotSpec(
                2, 1,
                subplot_spec=outer[flav_idx, orig_idx],
                height_ratios=[3, 1],
                hspace=0.05,
            )
            ax_top = fig.add_subplot(inner[0])
            ax_rat = fig.add_subplot(inner[1], sharex=ax_top)

            orig_mask = flat_valid & (flat_origin == orig_idx) & (flat_flav == flav_idx)
            vals      = flat_vals[orig_mask]

            edge = origin_edges[orig_idx]
            bins = np.linspace(-edge, edge, N_BINS_SYM + 1)
            ctrs = 0.5 * (bins[:-1] + bins[1:])

            colour = FLAVOUR_COLOURS[flav_idx]

            if vals.size >= 2:
                f_x, _   = np.histogram(vals,  bins=bins)
                # mirror: histogram of −x uses the same bins, giving f(−x) at each x
                f_mx, _  = np.histogram(-vals, bins=bins)

                ax_top.step(ctrs, f_x,  color=colour, linewidth=1.4,
                            label=r"$f(x)$")
                ax_top.step(ctrs, f_mx, color=colour, linewidth=1.4,
                            linestyle="--", alpha=0.75, label=r"$f(-x)$")

                ratio = np.where(f_mx > 0, f_x / f_mx, np.nan)
                ax_rat.step(ctrs, ratio, color=colour, linewidth=1.2)
                ax_rat.axhline(1.0, color="black", linewidth=0.7, linestyle="--")
                ax_rat.set_ylim(0, 2)

            # titles and labels
            ax_top.set_title(
                f"{flav_name}\n{orig_idx}: {ORIGIN_NAMES[orig_idx]}",
                fontsize=7, pad=3,
            )
            ax_top.tick_params(labelbottom=False, labelsize=6)
            ax_top.set_ylabel("Counts", fontsize=6)
            if vals.size >= 2 and orig_idx == 0:
                ax_top.legend(fontsize=6, loc="upper right")

            ax_rat.set_xlabel(xlabel, fontsize=6)
            ax_rat.set_ylabel("Ratio", fontsize=6)
            ax_rat.tick_params(labelsize=6)
            n_tracks = orig_mask.sum()
            ax_top.text(0.97, 0.95, f"n={n_tracks:,}",
                        transform=ax_top.transAxes,
                        ha="right", va="top", fontsize=5.5)

    safe = field.replace("/", "_")
    out  = os.path.join(PLOT_DIR, f"track_{safe}_symmetry_mirror.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def plot_asymmetry(field, xlabel, clip_pct):
    """
    A(x) = (f(x) − f(−x)) / (f(x) + f(−x))  for x > 0.
    A = 0 everywhere → symmetric.  A > 0 → excess of positive values.
    """
    flat_vals = var_arrays[field].ravel()

    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    fig.suptitle(
        f"{xlabel}  —  asymmetry  "
        r"$A(x) = \frac{f(x)-f(-x)}{f(x)+f(-x)}$",
        fontweight="bold", fontsize=12,
    )
    axes = axes.ravel()

    for orig_idx in range(N_ORIGINS):
        ax        = axes[orig_idx]
        orig_mask = flat_valid & (flat_origin == orig_idx)

        all_orig = flat_vals[orig_mask]
        if all_orig.size == 0:
            ax.set_visible(False)
            continue
        edge = np.percentile(np.abs(all_orig), clip_pct)
        bins = np.linspace(-edge, edge, N_BINS_SYM + 1)
        ctrs = 0.5 * (bins[:-1] + bins[1:])

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", zorder=0)

        for flav_idx, (flav_name, colour) in enumerate(zip(CLASS_NAMES, FLAVOUR_COLOURS)):
            mask = orig_mask & (flat_flav == flav_idx)
            vals = flat_vals[mask]
            if vals.size < 2:
                continue
            f_x,  _ = np.histogram(vals,  bins=bins)
            f_mx, _ = np.histogram(-vals, bins=bins)
            denom    = f_x + f_mx
            asym     = np.where(denom > 0, (f_x - f_mx) / denom, np.nan)
            err      = np.where(denom > 0, 1.0 / np.sqrt(denom + 1e-9), np.nan)

            label = flav_name if orig_idx == 0 else None
            ax.step(ctrs, asym, color=colour, linewidth=1.5, label=label)
            ax.fill_between(ctrs, asym - err, asym + err,
                            color=colour, alpha=0.15, step="mid")

        ax.set_title(f"{orig_idx}: {ORIGIN_NAMES[orig_idx]}", fontsize=10)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("Asymmetry $A(x)$", fontsize=8)
        ax.set_ylim(-1, 1)
        ax.tick_params(labelsize=7)
        if orig_idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    safe = field.replace("/", "_")
    out  = os.path.join(PLOT_DIR, f"track_{safe}_asymmetry.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


for field, xlabel, clip_pct in VARIABLES:
    plot_symmetry_mirror(field, xlabel, clip_pct)
    plot_asymmetry(field, xlabel, clip_pct)


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
