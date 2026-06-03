"""
Validation script for the trained Transformer jet flavour classifier.

Loads mc-flavtag-ttbar-small.h5, samples jets not used in training/testing,
runs the saved model on nominal inputs (soft flip is applied internally by
the model), and produces evaluation plots.

The model architecture must match transformer_jet_classifier.py exactly so
that load_state_dict succeeds. Flipping uses the same two-stage soft-flip
forward pass as training, with all model weights fixed (eval mode).
"""

import argparse
import hashlib
import json
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# ── defaults ───────────────────────────────────────────────────────────
_DEFAULTS = {
    "data_file":       "mc-flavtag-ttbar-small.h5",
    "model_file":      "models_cluster/transformer_jet_classifier_nomimal_btrackorigin_training.pt",
    "plot_dir":        "./transformer_results_val_nominal_boriginflip_origintrain_cluster/",
    "cache_dir":       ".track_cache_val_new/",
    "n_skip":          1_200_000,
    "n_val":           150_000,
    "top_k":           40,
    "batch_size":      400,
    "n_origins":       8,
    "flip_sharpness":  10,
    "flip_threshold":  0.5,
    "d_model":         32,
    "n_heads":         2,
    "n_layers":        2,
    "d_ffn":           64,
    "dropout":         0.1,
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
    # null means flip all tracks; list of ints selects specific origin classes
    # 0=Pileup 1=Fake 2=Primary 3=From b 4=From b->c 5=From c 6=From tau 7=Other secondary
    "flip_origins":      [3, 4],
    "flavour_to_label":  {"5": 0, "4": 1, "0": 2},
    "class_names":       ["b-jet", "c-jet", "light-jet"],
    "colours":           {"b-jet": "#1f77b4", "c-jet": "#ff7f0e", "light-jet": "#2ca02c"},
}

# ── args & config file ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Validate two-stage transformer jet classifier."
)
parser.add_argument("--config",    default=None,
                    help="Path to a JSON config file. Keys override the built-in defaults.")
parser.add_argument("--model",     default=None,
                    help="Override model_file from config.")
parser.add_argument("--plot-dir",  default=None,
                    help="Override plot_dir from config.")
args = parser.parse_args()

# merge: defaults → config file → CLI flags
cfg = dict(_DEFAULTS)
if args.config is not None:
    with open(args.config) as _f:
        _file_cfg = json.load(_f)
    # validate keys
    _unknown = set(_file_cfg) - set(_DEFAULTS)
    if _unknown:
        raise ValueError(f"Unknown config keys: {_unknown}")
    cfg.update(_file_cfg)
if args.model    is not None: cfg["model_file"] = args.model
if args.plot_dir is not None: cfg["plot_dir"]   = args.plot_dir

# ── unpack config into module-level names ──────────────────────────────
DATA_FILE       = cfg["data_file"]
MODEL_FILE      = cfg["model_file"]
PLOT_DIR        = cfg["plot_dir"]
CACHE_DIR       = cfg["cache_dir"]
N_SKIP          = cfg["n_skip"]
N_VAL           = cfg["n_val"]
TOP_K           = cfg["top_k"]
BATCH_SIZE      = cfg["batch_size"]
N_ORIGINS       = cfg["n_origins"]
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
FLIP_SHARPNESS  = cfg["flip_sharpness"]
FLIP_THRESHOLD  = cfg["flip_threshold"]
D_MODEL         = cfg["d_model"]
N_HEADS         = cfg["n_heads"]
N_LAYERS        = cfg["n_layers"]
D_FFN           = cfg["d_ffn"]
DROPOUT         = cfg["dropout"]
TRACK_FIELDS    = cfg["track_fields"]
FLIP_FIELDS     = cfg["flip_fields"]
FLIP_ORIGINS    = cfg["flip_origins"]          # None or list of ints
# JSON keys are strings; convert flavour ids back to int
FLAVOUR_TO_LABEL = {int(k): v for k, v in cfg["flavour_to_label"].items()}
CLASS_NAMES     = cfg["class_names"]
COLOURS         = cfg["colours"]

N_FEATS = len(TRACK_FIELDS)

os.makedirs(PLOT_DIR,   exist_ok=True)
os.makedirs(CACHE_DIR,  exist_ok=True)

# ── data loading ──────────────────────────────────────────────────────
def _cache_key(idx):
    h = hashlib.md5(idx.tobytes()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"tracks_{h}_nom.npz")


def load_tracks(path, idx):
    """Returns (N, K, F) features, (N, K) validity mask, (N,) labels, (N, K) origins.
    No hard flip is applied — the model's soft flip handles sign flipping.
    Origins are -1 for padded (invalid) tracks."""
    cp = _cache_key(idx)
    if os.path.exists(cp):
        d = np.load(cp)
        if "origins" in d:
            return d["X"], d["mask"], d["y"], d["origins"]

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

    sort_key = ip2d.copy()
    sort_key[~keep] = -np.inf
    order = np.argsort(-sort_key, axis=1)

    feats        = np.stack([arrs[fld] for fld in TRACK_FIELDS], axis=-1)
    topk_idx     = order[:, :TOP_K]
    rows         = np.arange(len(fidx))[:, None]
    topk_feat    = feats[rows, topk_idx]
    topk_valid   = keep[rows, topk_idx]
    topk_feat    = np.where(topk_valid[:, :, None], topk_feat, 0.0).astype(np.float32)
    topk_origin  = origin[rows, topk_idx].astype(np.int64)
    topk_origin[~topk_valid] = -1   # mask padding with ignore_index

    labels = np.array([FLAVOUR_TO_LABEL[v] for v in flavour_id[keep_jet]], dtype=np.int64)

    np.savez(cp, X=topk_feat, mask=topk_valid, y=labels, origins=topk_origin)
    return topk_feat, topk_valid, labels, topk_origin


class JetDataset(Dataset):
    def __init__(self, X, mask, y, origins):
        self.X       = torch.from_numpy(X)
        self.mask    = torch.from_numpy(mask)
        self.y       = torch.from_numpy(y)
        self.origins = torch.from_numpy(origins)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.mask[i], self.y[i], self.origins[i]


# ── model (must match transformer_jet_classifier.py exactly) ──────────
class JetTransformer(nn.Module):
    """
    Two-stage transformer with differentiable soft flipping.
    Must be identical to the definition in transformer_jet_classifier.py
    so that load_state_dict loads weights correctly.
      Stage 1 — reads original track features, predicts per-track origin (8 classes).
      Soft flip — flip_scale = -tanh(α*(p_flip - 0.5)):
                  p_flip → 0  ⟹  scale → +tanh(α/2) ≈ +1  (keep sign)
                  p_flip → 1  ⟹  scale → -tanh(α/2) ≈ -1  (flip sign)
                  p_flip = 0.5 ⟹ scale = 0 (ambiguous track zeroed)
      Stage 2 — reads soft-flipped features, classifies jet flavour.
    Returns: (jet_logits, jet_logits_pre_flip, track_logits_stage1)
    """
    def __init__(self, in_dim, d_model, n_heads, n_layers, d_ffn, dropout,
                 n_classes, n_origins, flip_feat_indices, flip_origin_indices,
                 flip_sharpness=10.0):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ffn,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Shared input projection and CLS token — both stages use identical jet classification weights
        self.input_proj = nn.Linear(in_dim, d_model)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Stage 1 head: per-track origin logits (unflipped encoder output)
        self.origin_head  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_origins))

        # Shared jet classifier: CLS token → jet flavour (used in both stages)
        self.classifier   = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

        self.register_buffer("flip_feat_mask",
            torch.zeros(in_dim, dtype=torch.bool).scatter_(
                0, torch.tensor(flip_feat_indices, dtype=torch.long), True))
        self.register_buffer("flip_origin_idx",
            torch.tensor(flip_origin_indices, dtype=torch.long))
        self.register_buffer("flip_sharpness",
            torch.tensor(flip_sharpness, dtype=torch.float32))

    def forward(self, x, mask):
        B = x.size(0)
        cls_valid            = torch.ones(B, 1, dtype=torch.bool, device=x.device)
        src_key_padding_mask = ~torch.cat([cls_valid, mask], dim=1)

        # Stage 1 — predict per-track origin probs + pre-flip jet logits
        h1           = self.input_proj(x)
        h1           = torch.cat([self.cls_token.expand(B, -1, -1), h1], dim=1)
        h1           = self.encoder(h1, src_key_padding_mask=src_key_padding_mask)
        track_logits   = self.origin_head(h1[:, 1:])
        jet_logits_pre = self.classifier(h1[:, 0])                           # (B, n_classes) pre-flip
        origin_probs = torch.softmax(track_logits, dim=-1)
        p_flip       = origin_probs[..., self.flip_origin_idx].sum(-1, keepdim=True)
        flip_scale   = torch.where(
            self.flip_feat_mask,
            -torch.tanh(self.flip_sharpness * (p_flip - FLIP_THRESHOLD)),
            torch.ones_like(p_flip),
        )
        x_soft = x * flip_scale

        # Stage 2 — same projection/CLS/encoder/classifier on soft-flipped features
        h2         = self.input_proj(x_soft)
        h2         = torch.cat([self.cls_token.expand(B, -1, -1), h2], dim=1)
        h2         = self.encoder(h2, src_key_padding_mask=src_key_padding_mask)
        jet_logits = self.classifier(h2[:, 0])

        return jet_logits, jet_logits_pre, track_logits


# ── load data (jets not seen during training/testing) ─────────────────
print("Loading validation data...")
rng = np.random.default_rng(42)
with h5py.File(DATA_FILE, "r") as f:
    n_total = f["jets"].shape[0]
all_idx = rng.permutation(n_total)
val_idx = np.sort(all_idx[N_SKIP:N_SKIP + N_VAL])

X_val, mask_val, y_val, origins_val = load_tracks(DATA_FILE, val_idx)
print(f"Validation jets: {len(y_val):,}  "
      f"(b={(y_val==0).sum():,}  c={(y_val==1).sum():,}  light={(y_val==2).sum():,})")

val_loader = DataLoader(JetDataset(X_val, mask_val, y_val, origins_val), batch_size=BATCH_SIZE)

# ── load model ────────────────────────────────────────────────────────
_flip_feat_idx   = [TRACK_FIELDS.index(f) for f in FLIP_FIELDS]
_flip_origin_idx = list(range(N_ORIGINS)) if FLIP_ORIGINS is None else list(FLIP_ORIGINS)

model = JetTransformer(N_FEATS, D_MODEL, N_HEADS, N_LAYERS, D_FFN, DROPOUT,
                       n_classes=3, n_origins=N_ORIGINS,
                       flip_feat_indices=_flip_feat_idx,
                       flip_origin_indices=_flip_origin_idx, flip_sharpness = FLIP_SHARPNESS).to(DEVICE)
model.load_state_dict(torch.load(MODEL_FILE, map_location=DEVICE))
model.eval()
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Loaded {MODEL_FILE}  ({n_params:,} parameters)")

# ── inference ─────────────────────────────────────────────────────────
def run_inference(loader):
    preds, trues, probs, probs_pre = [], [], [], []
    origin_preds, origin_trues = [], []
    with torch.no_grad():
        for X_b, mask_b, y_b, orig_b in loader:
            X_b, mask_b, y_b = X_b.to(DEVICE), mask_b.to(DEVICE), y_b.to(DEVICE)
            logits, logits_pre, track_logits_1 = model(X_b, mask_b)
            preds.append(logits.argmax(dim=1).cpu())
            trues.append(y_b.cpu())
            probs.append(torch.softmax(logits,     dim=1).cpu())  # Stage 2 post-flip
            probs_pre.append(torch.softmax(logits_pre, dim=1).cpu())  # Stage 1 pre-flip
            origin_preds.append(track_logits_1.argmax(dim=-1).cpu())  # (B, K) — Stage 1
            origin_trues.append(orig_b)                                # (B, K)
    # flatten tracks, drop padding (true origin == -1)
    op = torch.cat(origin_preds).numpy().ravel()
    ot = torch.cat(origin_trues).numpy().ravel()
    valid_tracks = ot >= 0
    return (torch.cat(preds).numpy(),
            torch.cat(trues).numpy(),
            torch.cat(probs).numpy(),
            torch.cat(probs_pre).numpy(),
            op[valid_tracks],
            ot[valid_tracks])

all_preds, all_true, all_probs, all_probs_pre, origin_preds, origin_true = run_inference(val_loader)

acc = (all_preds == all_true).mean()
print(f"\nAccuracy: {acc:.4f}")
print(classification_report(all_true, all_preds, target_names=CLASS_NAMES))
print("Confusion matrix:")
print(confusion_matrix(all_true, all_preds))

def make_disc(probs):
    pb, pc, pu = probs[:, 0], probs[:, 1], probs[:, 2]
    return np.log(pb / (0.2 * pc + 0.8 * pu + 1e-10))

disc     = make_disc(all_probs)      # Stage 2 post-flip
disc_pre = make_disc(all_probs_pre)  # Stage 1 pre-flip

CASES = [
    ("Post-flip (Stage 2)", all_probs,     disc),
    ("Pre-flip  (Stage 1)", all_probs_pre, disc_pre),
]
LINESTYLES = {"Post-flip (Stage 2)": "-", "Pre-flip  (Stage 1)": "--"}
ROC_COLOURS = {"Post-flip (Stage 2)": "#1f77b4", "Pre-flip  (Stage 1)": "#d62728"}

# ── plot: input variables ─────────────────────────────────────────────
n_cols = min(N_FEATS, 4)
n_rows = (N_FEATS + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = np.array(axes).ravel()
fig.suptitle("Input variables by jet flavour (validation sample)", fontweight="bold")
tracks_flat = X_val.reshape(-1, N_FEATS)
labels_rep  = np.repeat(all_true, TOP_K)
nonzero     = mask_val.ravel()
for fi, fld in enumerate(TRACK_FIELDS):
    col = tracks_flat[:, fi]
    valid_col = col[nonzero]
    clip = np.percentile(np.abs(valid_col), 99) if valid_col.size else 1.0
    for cls_idx, name in enumerate(CLASS_NAMES):
        m = (labels_rep == cls_idx) & nonzero
        axes[fi].hist(col[m], bins=80, range=(-clip, clip),
                      histtype="step", label=name, color=COLOURS[name],
                      linewidth=1.5, density=True)
    axes[fi].set_title(fld, fontsize=8)
    axes[fi].set_xlabel(fld, fontsize=7)
    axes[fi].set_ylabel("Density", fontsize=7)
    if fi == 0:
        axes[fi].legend(fontsize=7)
for ax in axes[N_FEATS:]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(PLOT_DIR + "val_input_variables.png", dpi=150, bbox_inches="tight")
print("Saved val_input_variables.png")

# ── plot: track origin confusion matrix ──────────────────────────────
ORIGIN_NAMES = ["Pileup", "Fake", "Primary", "From b", "From b→c", "From c", "From τ", "Other sec."]
present      = sorted(np.unique(origin_true))   # only classes that actually appear
labels_pres  = [ORIGIN_NAMES[i] for i in present]

cm_orig = confusion_matrix(origin_true, origin_preds, labels=present, normalize="true")
fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle("Track origin confusion matrix (normalised, Stage 2)", fontweight="bold")
im = ax.imshow(cm_orig, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(present))); ax.set_yticks(range(len(present)))
ax.set_xticklabels(labels_pres, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(labels_pres, fontsize=8)
ax.set_xlabel("Predicted origin"); ax.set_ylabel("True origin")
for i in range(len(present)):
    for j in range(len(present)):
        ax.text(j, i, f"{cm_orig[i,j]:.2f}", ha="center", va="center", fontsize=7,
                color="white" if cm_orig[i,j] > 0.5 else "black")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(PLOT_DIR + "val_origin_confusion.png", dpi=150, bbox_inches="tight")
print("Saved val_origin_confusion.png")

# ── plot: confusion matrix ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
fig.suptitle("Confusion matrix (normalised)", fontweight="bold")
cm = confusion_matrix(all_true, all_preds, normalize="true")
im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 1, 2])
ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                color="white" if cm[i,j] > 0.5 else "black")
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(PLOT_DIR + "val_confusion_matrix.png", dpi=150, bbox_inches="tight")
print("Saved val_confusion_matrix.png")

# ── plot: output probabilities ────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Transformer output probabilities by true flavour", fontweight="bold")
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    ax = axes[cls_idx]
    for true_idx, true_name in enumerate(CLASS_NAMES):
        mask = all_true == true_idx
        ax.hist(all_probs[mask, cls_idx], bins=50, range=(0, 1), histtype="step",
                label=true_name, color=COLOURS[true_name], linewidth=1.5, density=True)
    ax.set_title(f"P({cls_name})")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Density")
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig(PLOT_DIR + "val_output_probs.png", dpi=150, bbox_inches="tight")
print("Saved val_output_probs.png")

# ── plot: discriminant ────────────────────────────────────────────────
finite    = np.isfinite(disc)
clip_disc = np.percentile(np.abs(disc[finite]), 99)

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle(r"$\log(p_b\,/\,(0.2\,p_c + 0.8\,p_u))$", fontweight="bold")
for true_idx, true_name in enumerate(CLASS_NAMES):
    m = (all_true == true_idx) & finite
    ax.hist(disc[m], bins=80, range=(-clip_disc, clip_disc), histtype="step",
            label=true_name, color=COLOURS[true_name], linewidth=1.5, density=True)
ax.set_xlabel(r"$\log(p_b\,/\,(0.2\,p_c + 0.8\,p_u))$")
ax.set_ylabel("Density")
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR + "val_discriminant.png", dpi=150, bbox_inches="tight")
print("Saved val_discriminant.png")

# ── plot: ROC curves ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(r"ROC curves — $\log(p_b\,/\,(0.2\,p_c + 0.8\,p_u))$", fontweight="bold")
for ax, (bkg_idx, bkg_name) in zip(axes, [(1, "c-jet"), (2, "light-jet")]):
    m      = (all_true == 0) | (all_true == bkg_idx)
    scores = (all_true[m] == 0).astype(int)
    score  = disc[m]
    finite = np.isfinite(score)
    fpr, tpr, _ = roc_curve(scores[finite], score[finite])
    ax.plot(tpr, fpr, color="#1f77b4", linewidth=1.5, label=f"AUC={auc(fpr, tpr):.3f}")
    ax.set_xlabel("b-jet efficiency (TPR)")
    ax.set_ylabel(f"{bkg_name} rate (FPR)")
    ax.set_title(f"b vs {bkg_name}")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(PLOT_DIR + "val_roc.png", dpi=150, bbox_inches="tight")
print("Saved val_roc.png")

# ── plot: discriminant overlay — post-flip vs pre-flip ────────────────
all_disc_cat = np.concatenate([disc, disc_pre])
finite_all   = np.isfinite(all_disc_cat)
clip_disc    = np.percentile(np.abs(all_disc_cat[finite_all]), 99)
disc_bins    = np.linspace(-clip_disc, clip_disc, 81)

fig = plt.figure(figsize=(15, 8))
fig.suptitle(r"$\log(p_b\,/\,(0.2\,p_c + 0.8\,p_u))$: post-flip Stage 2 (solid) vs pre-flip Stage 1 (dashed)",
             fontweight="bold")
gs = fig.add_gridspec(2, 3, hspace=0.08, wspace=0.35, height_ratios=[3, 1])

for cls_idx, cls_name in enumerate(CLASS_NAMES):
    ax_main  = fig.add_subplot(gs[0, cls_idx])
    ax_ratio = fig.add_subplot(gs[1, cls_idx], sharex=ax_main)

    counts = {}
    for label, probs_c, d in CASES:
        finite = np.isfinite(d)
        m      = (all_true == cls_idx) & finite
        h, _   = np.histogram(d[m], bins=disc_bins, density=True)
        counts[label] = h
        ax_main.stairs(h, disc_bins, color=COLOURS[cls_name],
                       linestyle=LINESTYLES[label], linewidth=1.5, label=label)

    ax_main.set_title(f"True {cls_name}")
    ax_main.set_ylabel("Density")
    ax_main.legend(fontsize=8)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    h_post, h_pre = counts["Post-flip (Stage 2)"], counts["Pre-flip  (Stage 1)"]
    valid_bin = h_post > 0
    ratio     = np.where(valid_bin, h_pre / np.where(valid_bin, h_post, 1), np.nan)
    ax_ratio.stairs(ratio, disc_bins, color=COLOURS[cls_name], linewidth=1.2)
    ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax_ratio.set_ylim(0.5, 2.0)
    ax_ratio.set_ylabel("Pre/Post", fontsize=7)
    ax_ratio.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax_ratio.tick_params(labelsize=7)
    ax_ratio.set_xlabel(r"$\log(p_b\,/\,(0.2\,p_c + 0.8\,p_u))$")

plt.savefig(PLOT_DIR + "val_discriminant_overlay.png", dpi=150, bbox_inches="tight")
print("Saved val_discriminant_overlay.png")

# ── plot: 2D scatter — pre-flip vs post-flip discriminant (light jets) ───
light_mask  = (all_true == 2) & np.isfinite(disc) & np.isfinite(disc_pre)
x_scatter   = disc_pre[light_mask]   # Stage 1 pre-flip
y_scatter   = disc[light_mask]       # Stage 2 post-flip

# clip to 99th percentile for display
clip_x = np.percentile(np.abs(x_scatter), 99.9)
clip_y = np.percentile(np.abs(y_scatter), 99.9)
lim    = max(clip_x, clip_y)

# subsample if large (scatter is slow for >50k points)
MAX_PTS = 50_000
if len(x_scatter) > MAX_PTS:
    rng_sc  = np.random.default_rng(0)
    sel     = rng_sc.choice(len(x_scatter), MAX_PTS, replace=False)
    x_sc, y_sc = x_scatter[sel], y_scatter[sel]
else:
    x_sc, y_sc = x_scatter, y_scatter

from scipy.stats import gaussian_kde

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x_sc, y_sc, s=2, alpha=0.8, color=COLOURS["light-jet"], linewidths=0)

# 90% contour via KDE on the (subsampled) scatter points
_kde        = gaussian_kde(np.vstack([x_sc, y_sc]))
_grid_pts   = 200
_gx         = np.linspace(-lim, lim, _grid_pts)
_gy         = np.linspace(-lim, lim, _grid_pts)
_XX, _YY    = np.meshgrid(_gx, _gy)
_Z          = _kde(np.vstack([_XX.ravel(), _YY.ravel()])).reshape(_grid_pts, _grid_pts)
# find density threshold enclosing 90% of the probability mass
_z_flat     = np.sort(_Z.ravel())[::-1]
_cumsum     = np.cumsum(_z_flat) / _z_flat.sum()
_threshold  = _z_flat[np.searchsorted(_cumsum, 0.99)]
ax.contour(_XX, _YY, _Z, levels=[_threshold], colors=["#d62728"], linewidths=1.5,
           linestyles="-")
ax.plot([], [], color="#d62728", linewidth=1.5, label="99% contour")

ax.axline((0, 0), slope=1, color="black", linewidth=0.8, linestyle="--", label="y = x")
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel(r"Pre-flip discriminant (Stage 1)  $\log(p_b\,/\,(0.2\,p_c+0.8\,p_u))$")
ax.set_ylabel(r"Post-flip discriminant (Stage 2)  $\log(p_b\,/\,(0.2\,p_c+0.8\,p_u))$")
ax.set_title("Light jets: pre-flip vs post-flip discriminant", fontweight="bold")
ax.legend(fontsize=8)
ax.set_aspect("equal")
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR + "val_disc_scatter_light.png", dpi=150, bbox_inches="tight")
print("Saved val_disc_scatter_light.png")

# ── plot: 2D scatter — tail (top 10% pre-flip b-score) light jets ─────
tail10_thresh  = np.percentile(x_scatter, 90)   # top 10% by pre-flip disc
tail10_sel     = x_scatter >= tail10_thresh
x_tail, y_tail = x_scatter[tail10_sel], y_scatter[tail10_sel]

# axis bounds: use 1st–99th percentile of each axis to focus on the data
x_lo_t = np.percentile(x_tail, 1);   x_hi_t = np.percentile(x_tail, 99)
y_lo_t = np.percentile(y_tail, 1);   y_hi_t = np.percentile(y_tail, 99)
pad_x  = 0.05 * (x_hi_t - x_lo_t);  pad_y  = 0.05 * (y_hi_t - y_lo_t)
xlim_t = (x_lo_t - pad_x, x_hi_t + pad_x)
ylim_t = (y_lo_t - pad_y, y_hi_t + pad_y)

MAX_PTS_T = 50_000
if len(x_tail) > MAX_PTS_T:
    rng_t        = np.random.default_rng(1)
    sel_t        = rng_t.choice(len(x_tail), MAX_PTS_T, replace=False)
    x_sc_t, y_sc_t = x_tail[sel_t], y_tail[sel_t]
else:
    x_sc_t, y_sc_t = x_tail, y_tail

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(x_sc_t, y_sc_t, s=2, alpha=0.8, color=COLOURS["light-jet"], linewidths=0)

# 99% contour via KDE on the tail points
_kde_t       = gaussian_kde(np.vstack([x_sc_t, y_sc_t]))
_gx_t        = np.linspace(xlim_t[0], xlim_t[1], _grid_pts)
_gy_t        = np.linspace(ylim_t[0], ylim_t[1], _grid_pts)
_XX_t, _YY_t = np.meshgrid(_gx_t, _gy_t)
_Z_t         = _kde_t(np.vstack([_XX_t.ravel(), _YY_t.ravel()])).reshape(_grid_pts, _grid_pts)
_z_flat_t    = np.sort(_Z_t.ravel())[::-1]
_cumsum_t    = np.cumsum(_z_flat_t) / _z_flat_t.sum()
_thresh_t    = _z_flat_t[np.searchsorted(_cumsum_t, 0.99)]
ax.contour(_XX_t, _YY_t, _Z_t, levels=[_thresh_t], colors=["#d62728"], linewidths=1.5,
           linestyles="-")
ax.plot([], [], color="#d62728", linewidth=1.5, label="99% contour")

ax.axline((0, 0), slope=1, color="black", linewidth=0.8, linestyle="--", label="y = x")
ax.set_xlim(*xlim_t)
ax.set_ylim(*ylim_t)
ax.set_xlabel(r"Pre-flip discriminant (Stage 1)  $\log(p_b\,/\,(0.2\,p_c+0.8\,p_u))$")
ax.set_ylabel(r"Post-flip discriminant (Stage 2)  $\log(p_b\,/\,(0.2\,p_c+0.8\,p_u))$")
ax.set_title("Light jets (top 10% pre-flip tail): pre-flip vs post-flip discriminant",
             fontweight="bold")
ax.legend(fontsize=8)
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR + "val_disc_scatter_light_tail10.png", dpi=150, bbox_inches="tight")
print("Saved val_disc_scatter_light_tail10.png")

# ── zoom-in tail plots: 5%, 1%, 0.1% ─────────────────────────────────
light_post = disc[light_mask]
light_pre  = disc_pre[light_mask]

for pct, pct_label, fname, n_bins in [
    (95,   "top 5%",   "val_disc_tail_zoom_light.png",      60),
    (99,   "top 1%",   "val_disc_tail_zoom_light_1pct.png", 20),
    (99.9, "top 0.1%", "val_disc_tail_zoom_light_01pct.png", 10),
]:
    tail_lo   = np.percentile(light_post, pct)
    tail_hi   = max(np.percentile(light_post, 99.99), np.percentile(light_pre, 99.99))
    tail_bins = np.linspace(tail_lo, tail_hi, n_bins)

    fig, (ax_main, ax_ratio) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    fig.suptitle(f"Light jets — high-tail zoom ({pct_label}): pre-flip vs post-flip",
                 fontweight="bold")

    counts_t = {}
    for d_arr, label, ls in [
        (light_post, "Post-flip (Stage 2)", "-"),
        (light_pre,  "Pre-flip  (Stage 1)", "--"),
    ]:
        h, _ = np.histogram(d_arr, bins=tail_bins)
        counts_t[label] = h
        ax_main.stairs(h, tail_bins, linestyle=ls, linewidth=1.8,
                       color=COLOURS["light-jet"], label=label,
                       alpha=(1.0 if ls == "-" else 0.6))

    ax_main.set_ylabel("Counts")
    ax_main.legend(fontsize=9)
    ax_main.grid(True, linestyle="--", alpha=0.3)
    plt.setp(ax_main.get_xticklabels(), visible=False)

    h_post_t = counts_t["Post-flip (Stage 2)"]
    h_pre_t  = counts_t["Pre-flip  (Stage 1)"]
    valid_t  = h_post_t > 0
    ratio_t  = np.where(valid_t, h_pre_t / np.where(valid_t, h_post_t, 1), np.nan)
    ax_ratio.stairs(ratio_t, tail_bins, color=COLOURS["light-jet"], linewidth=1.2)
    ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax_ratio.set_ylim(0.5, 2.0)
    ax_ratio.set_ylabel("Pre/Post", fontsize=8)
    ax_ratio.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax_ratio.tick_params(labelsize=8)
    ax_ratio.grid(True, linestyle="--", alpha=0.3)
    ax_ratio.set_xlabel(r"$\log(p_b\,/\,(0.2\,p_c + 0.8\,p_u))$")

    plt.savefig(PLOT_DIR + fname, dpi=150, bbox_inches="tight")
    print(f"Saved {fname}")

# ── plot: probability overlay — 3×3 with ratio panels ────────────────
BINS  = 50
RANGE = (0, 1)
bin_edges = np.linspace(RANGE[0], RANGE[1], BINS + 1)

fig = plt.figure(figsize=(15, 18))
fig.suptitle("Output probabilities: post-flip Stage 2 (solid) vs pre-flip Stage 1 (dashed)\n"
             "Ratio: pre / post", fontweight="bold")
gs = fig.add_gridspec(6, 3, hspace=0.08, wspace=0.35,
                      height_ratios=[3, 1, 3, 1, 3, 1])

for true_idx, true_name in enumerate(CLASS_NAMES):
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        ax_main  = fig.add_subplot(gs[true_idx * 2,     cls_idx])
        ax_ratio = fig.add_subplot(gs[true_idx * 2 + 1, cls_idx], sharex=ax_main)

        counts = {}
        for label, probs_c, _ in CASES:
            m = all_true == true_idx
            h, _ = np.histogram(probs_c[m, cls_idx], bins=bin_edges, density=True)
            counts[label] = h
            ax_main.stairs(h, bin_edges, color=COLOURS[true_name],
                           linestyle=LINESTYLES[label], linewidth=1.5, label=label)

        h_post, h_pre = counts["Post-flip (Stage 2)"], counts["Pre-flip  (Stage 1)"]
        valid_bin = h_post > 0
        ratio     = np.where(valid_bin, h_pre / np.where(valid_bin, h_post, 1), np.nan)
        ax_ratio.stairs(ratio, bin_edges, color=COLOURS[true_name], linewidth=1.2)
        ax_ratio.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
        ax_ratio.set_ylim(0.5, 2.0)
        ax_ratio.set_ylabel("Pre/Post", fontsize=7)
        ax_ratio.yaxis.set_major_locator(plt.MultipleLocator(0.25))
        ax_ratio.tick_params(labelsize=7)

        ax_main.set_title(f"True {true_name} — P({cls_name})", fontsize=9)
        ax_main.set_ylabel("Density")
        ax_main.legend(fontsize=8)
        plt.setp(ax_main.get_xticklabels(), visible=False)

        if true_idx == 2:
            ax_ratio.set_xlabel("Probability")

plt.savefig(PLOT_DIR + "val_prob_overlay.png", dpi=150, bbox_inches="tight")
print("Saved val_prob_overlay.png")

# ── plot: ROC overlay — post-flip vs pre-flip ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(r"ROC curves — post-flip Stage 2 vs pre-flip Stage 1", fontweight="bold")
for ax, (bkg_idx, bkg_name) in zip(axes, [(1, "c-jet"), (2, "light-jet")]):
    for label, probs_c, d in CASES:
        m      = (all_true == 0) | (all_true == bkg_idx)
        scores = (all_true[m] == 0).astype(int)
        score  = d[m]
        finite = np.isfinite(score)
        fpr, tpr, _ = roc_curve(scores[finite], score[finite])
        ax.plot(tpr, fpr, color=ROC_COLOURS[label], linewidth=1.5,
                label=f"{label}  AUC={auc(fpr, tpr):.3f}",
                linestyle=LINESTYLES[label])
    ax.set_xlabel("b-jet efficiency (TPR)")
    ax.set_ylabel(f"{bkg_name} rate (FPR)")
    ax.set_title(f"b vs {bkg_name}")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(PLOT_DIR + "val_roc_overlay.png", dpi=150, bbox_inches="tight")
print("Saved val_roc_overlay.png")

# ── table: mis-ID rates at fixed b-jet efficiencies ──────────────────
B_EFFS = [0.90, 0.70, 0.50]

# Thresholds are always derived from the pre-flip (Stage 1) b-jet discriminant
_pre_b_disc = disc_pre[all_true == 0]
_pre_b_disc = _pre_b_disc[np.isfinite(_pre_b_disc)]
THRESHOLDS  = {eff: np.percentile(_pre_b_disc, 100.0 * (1.0 - eff)) for eff in B_EFFS}

def stats_at_thresholds(disc_arr, true_arr, thresholds):
    """Return (b_eff, light_misid, charm_misid) at each threshold."""
    b_disc     = disc_arr[(true_arr == 0) & np.isfinite(disc_arr)]
    light_disc = disc_arr[(true_arr == 2) & np.isfinite(disc_arr)]
    charm_disc = disc_arr[(true_arr == 1) & np.isfinite(disc_arr)]
    out = []
    for thr in thresholds.values():
        out.append((
            (b_disc     >= thr).mean(),
            (light_disc >= thr).mean(),
            (charm_disc >= thr).mean(),
        ))
    return out  # list of (b_eff, light_misid, charm_misid) per working point

pre_stats  = stats_at_thresholds(disc_pre, all_true, THRESHOLDS)
post_stats = stats_at_thresholds(disc,     all_true, THRESHOLDS)

# one row per working point
col_labels = [
    "Pre-flip b-eff", "Post-flip b-eff",
    "Pre-flip light mis-ID", "Post-flip light mis-ID",
    "Pre-flip charm mis-ID", "Post-flip charm mis-ID",
]
rows = []
for (pre_beff, pre_lr, pre_cr), (post_beff, post_lr, post_cr) in zip(pre_stats, post_stats):
    rows.append((pre_beff, post_beff, pre_lr, post_lr, pre_cr, post_cr))

# print to terminal
w = 16
header = (f"{'Pre b-eff':>{w}}  {'Post b-eff':>{w}}"
          f"  {'Pre light':>{w}}  {'Post light':>{w}}"
          f"  {'Pre charm':>{w}}  {'Post charm':>{w}}")
print("\n" + "─" * len(header))
print(header)
print("─" * len(header))
for r in rows:
    print("  ".join(f"{v:>{w}.4%}" for v in r))
print("─" * len(header))

# save as a matplotlib table figure
fig, ax = plt.subplots(figsize=(14, 0.65 * (len(rows) + 2)))
ax.axis("off")
cell_text = [[f"{v:.4%}" for v in r] for r in rows]
row_cols  = [["#f0f0f0" if i % 2 == 0 else "white"] * len(col_labels)
             for i in range(len(rows))]

tbl = ax.table(cellText=cell_text, colLabels=col_labels,
               cellColours=row_cols, loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.8)
fig.suptitle(
    "Mis-identification rates at fixed b-jet efficiency\n"
    "(thresholds set by pre-flip Stage 1 discriminant)",
    fontweight="bold", y=0.98,
)
plt.tight_layout()
plt.savefig(PLOT_DIR + "val_misid_table.png", dpi=150, bbox_inches="tight")
print("Saved val_misid_table.png")

# save as a standalone compilable LaTeX file
_tex_col_spec = "cc|cc|cc"
_tex_header   = (
    r"\textbf{Pre-flip $b$-eff} & \textbf{Post-flip $b$-eff} & "
    r"\textbf{Pre-flip light} & \textbf{Post-flip light} & "
    r"\textbf{Pre-flip charm} & \textbf{Post-flip charm}"
)
_tex_rows = []
for r in rows:
    _tex_rows.append(" & ".join(f"{v:.4%}".replace("%", r"\%") for v in r) + r" \\")

_tex = r"""\documentclass{article}
\usepackage{booktabs}
\usepackage{geometry}
\geometry{margin=1in}
\begin{document}

\begin{table}[ht]
  \centering
  \caption{Mis-identification rates at fixed $b$-jet tagging efficiency.
           Thresholds are set by the pre-flip (Stage~1) discriminant.
           Columns are paired by jet flavour.}
  \begin{tabular}{""" + _tex_col_spec + r"""}
    \toprule
    """ + _tex_header + r""" \\
    \midrule
""" + "\n".join("    " + row for row in _tex_rows) + r"""
    \bottomrule
  \end{tabular}
\end{table}

\end{document}
"""

_tex_path = PLOT_DIR + "val_misid_table.tex"
with open(_tex_path, "w") as _f:
    _f.write(_tex)
print(f"Saved val_misid_table.tex  (compile with: pdflatex {_tex_path})")

# compile to PDF
import subprocess
_result = subprocess.run(
    ["pdflatex", "-interaction=nonstopmode", "-output-directory", PLOT_DIR,
     _tex_path],
    capture_output=True, text=True,
)
if _result.returncode == 0:
    print("Compiled val_misid_table.pdf")
else:
    print("pdflatex failed — check val_misid_table.log for details")
    print(_result.stdout[-2000:])   # last 2000 chars of log is usually enough

print(f"\nAll plots saved to {PLOT_DIR}")
