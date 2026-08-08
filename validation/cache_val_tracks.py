"""
Build and save nominal + flipped validation track caches for future studies.

Usage:
    python cache_val_tracks.py [--config path/to/config.json]
                               [--plot-dir path/to/override/plot_dir]

The two cache files written are:
    <CACHE_DIR>/tracks_<hash>_nom.npz          -- nominal (already saved by validate_transformer.py)
    <CACHE_DIR>/tracks_<hash>_flip<origins>.npz -- hard-flipped (new)
"""

import argparse
import hashlib
import json
import os
import h5py
import numpy as np

# ── defaults ───────────────────────────────────────────────────────────
_DEFAULTS = {
    "data_file":      "/large-data/transformer/jetset/93940/mc-flavtag-ttbar-large.h5",
    "val_cache_dir":  "val_cache_allflip/",
    "n_skip":         12_000_000,
    "n_val":          1_200_000,
    "top_k":          40,
    "track_fields": [
        "qOverP", "deta", "dphi", "d0", "z0SinTheta",
        "qOverPUncertainty", "thetaUncertainty", "phiUncertainty",
        "lifetimeSignedD0Significance", "lifetimeSignedZ0SinThetaSignificance",
        "numberOfPixelHits", "numberOfSCTHits",
        "numberOfInnermostPixelLayerHits", "numberOfNextToInnermostPixelLayerHits",
        "numberOfInnermostPixelLayerSharedHits", "numberOfInnermostPixelLayerSplitHits",
        "numberOfPixelSharedHits", "numberOfPixelSplitHits", "numberOfSCTSharedHits",
    ],
    "flip_fields": [
        "lifetimeSignedD0Significance", "lifetimeSignedZ0SinThetaSignificance",
        "d0", "z0SinTheta",
    ],
    "flip_origins":     [0,1,2,3,4,5,6,7],
    "flavour_to_label": {"5": 0, "4": 1, "0": 2},
}

parser = argparse.ArgumentParser(description="Cache nominal and flipped validation tracks.")
parser.add_argument("--config",   default=None, help="JSON config file.")
parser.add_argument("--plot-dir", default=None, help="Unused; accepted for compatibility.")
args = parser.parse_args()

cfg = dict(_DEFAULTS)
if args.config is not None:
    with open(args.config) as _f:
        _file_cfg = json.load(_f)
    cfg.update({k: v for k, v in _file_cfg.items() if k in _DEFAULTS})

DATA_FILE        = cfg["data_file"]
CACHE_DIR        = cfg["val_cache_dir"]
N_SKIP           = cfg["n_skip"]
N_VAL            = cfg["n_val"]
TOP_K            = cfg["top_k"]
TRACK_FIELDS     = cfg["track_fields"]
FLIP_FIELDS      = cfg["flip_fields"]
FLIP_ORIGINS     = cfg["flip_origins"]
FLAVOUR_TO_LABEL = {int(k): v for k, v in cfg["flavour_to_label"].items()}

os.makedirs(CACHE_DIR, exist_ok=True)

# ── helpers ────────────────────────────────────────────────────────────
def _cache_key(idx, flip):
    origins_tag = "all" if FLIP_ORIGINS is None else "".join(str(o) for o in sorted(FLIP_ORIGINS))
    h   = hashlib.md5(idx.tobytes()).hexdigest()[:12]
    tag = f"flip{origins_tag}" if flip else "nom"
    return os.path.join(CACHE_DIR, f"tracks_{h}_{tag}.npz")


def build_and_save(path, idx, flip):
    cp = _cache_key(idx, flip)
    if os.path.exists(cp):
        print(f"  already exists, skipping: {cp}")
        return

    with h5py.File(path, "r") as f:
        flavour_id = f["jets"]["HadronConeExclTruthLabelID"][idx]
        keep_jet   = np.isin(flavour_id, list(FLAVOUR_TO_LABEL.keys()))
        fidx       = idx[keep_jet]

        valid  = f["tracks"]["valid"][fidx]
        d0     = f["tracks"]["d0"][fidx].astype(np.float32)
        ip2d   = f["tracks"]["lifetimeSignedD0Significance"][fidx].astype(np.float32)
        origin = f["tracks"]["GN2v01_trackOrigin"][fidx].astype(np.int8)
        arrs   = {fld: f["tracks"][fld][fidx].astype(np.float32) for fld in TRACK_FIELDS}

    keep = valid & (np.abs(d0) < 3.5)

    if flip:
        flip_mask = (np.ones_like(origin, dtype=bool) if FLIP_ORIGINS is None
                     else np.isin(origin, FLIP_ORIGINS))
        ip2d_sort = ip2d.copy()
        ip2d_sort[flip_mask] = -ip2d_sort[flip_mask]
    else:
        flip_mask = np.zeros_like(origin, dtype=bool)
        ip2d_sort = ip2d

    sort_key = ip2d_sort.copy()
    sort_key[~keep] = -np.inf
    order = np.argsort(-sort_key, axis=1)

    feat_list = []
    for fld in TRACK_FIELDS:
        arr = arrs[fld].copy()
        if flip and fld in FLIP_FIELDS:
            arr[flip_mask] = -arr[flip_mask]
        feat_list.append(arr)
    feats = np.stack(feat_list, axis=-1)

    topk_idx    = order[:, :TOP_K]
    rows        = np.arange(len(fidx))[:, None]
    topk_feat   = feats[rows, topk_idx]
    topk_valid  = keep[rows, topk_idx]
    topk_feat   = np.where(topk_valid[:, :, None], topk_feat, 0.0).astype(np.float32)
    topk_origin = origin[rows, topk_idx].astype(np.int64)
    topk_origin[~topk_valid] = -1

    labels = np.array([FLAVOUR_TO_LABEL[v] for v in flavour_id[keep_jet]], dtype=np.int64)

    np.savez(cp, X=topk_feat, mask=topk_valid, y=labels, origins=topk_origin)
    print(f"  saved: {cp}  ({len(labels):,} jets)")


# ── build caches ───────────────────────────────────────────────────────
rng = np.random.default_rng(42)
with h5py.File(DATA_FILE, "r") as f:
    n_total = f["jets"].shape[0]
all_idx = rng.permutation(n_total)
val_idx = np.sort(all_idx[-N_VAL:])

print(f"Validation index: {len(val_idx):,} jets (skip={N_SKIP:,})")

print("Building nominal cache...")
build_and_save(DATA_FILE, val_idx, flip=False)

print("Building flipped cache...")
build_and_save(DATA_FILE, val_idx, flip=True)

print("Done.")
