"""
Transformer jet flavour classifier.
Based on gnn_jet_classifier.py — identical data pipeline, balanced training,
and symmetry loss. Only the model is changed: a CLS-token transformer encoder
instead of fully-connected message passing.
"""
import argparse
import hashlib
import json
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# ── defaults ───────────────────────────────────────────────────────────
_DEFAULTS = {
    # shared with validation script
    "top_k":          40,
    "batch_size":     800,
    "n_origins":      8,
    "flip_sharpness": 10.0,
    "flip_threshold": 0.5,
    "d_model":        32,
    "n_heads":        2,
    "n_layers":       2,
    "d_ffn":          64,
    "dropout":        0.1,
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
    # training-only
    "train_file":      "/large-data/transformer/jetset/93940/mc-flavtag-ttbar-small.h5",
    "n_train":         1_200_000,
    "n_test":          600_000,
    "epochs":          100,
    "lr":              1e-3,
    "num_workers":     8,
    "lambda_sym":      100,
    "lambda_orig":     1,
    "b_ratio":         0.0,
    "model_name":      "transformer_jet_classifier_nomimal_btrackorigin_training.pt",
    "train_plot_dir":  "./transformer_results_nomimal_btrackorigin_training/",
    "train_cache_dir": ".track_cache_new/",
}

# ── args & config file ─────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Train two-stage transformer jet classifier."
)
parser.add_argument("--config", default=None,
                    help="Path to a JSON config file. Keys override the built-in defaults.")
args = parser.parse_args()

cfg = dict(_DEFAULTS)
if args.config is not None:
    with open(args.config) as _f:
        _file_cfg = json.load(_f)
    _unknown = set(_file_cfg) - set(_DEFAULTS)
    if _unknown:
        raise ValueError(f"Unknown config keys: {_unknown}")
    cfg.update(_file_cfg)

# ── unpack config into module-level names ──────────────────────────────
TRAIN_FILE      = cfg["train_file"]
N_TRAIN         = cfg["n_train"]
N_TEST          = cfg["n_test"]
BATCH_SIZE      = cfg["batch_size"]
EPOCHS          = cfg["epochs"]
LR              = cfg["lr"]
NUM_WORKERS     = cfg["num_workers"]
TOP_K           = cfg["top_k"]
LAMBDA_SYM      = cfg["lambda_sym"]
LAMBDA_ORIG     = cfg["lambda_orig"]
B_RATIO         = cfg["b_ratio"]
FLIP_SHARPNESS  = cfg["flip_sharpness"]
FLIP_THRESHOLD  = cfg["flip_threshold"]
N_ORIGINS       = cfg["n_origins"]
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME      = cfg["model_name"]
PLOT_DIR        = cfg["train_plot_dir"]
CACHE_DIR       = cfg["train_cache_dir"]
D_MODEL         = cfg["d_model"]
N_HEADS         = cfg["n_heads"]
N_LAYERS        = cfg["n_layers"]
D_FFN           = cfg["d_ffn"]
DROPOUT         = cfg["dropout"]
TRACK_FIELDS    = cfg["track_fields"]
FLIP_FIELDS     = cfg["flip_fields"]
FLIP_ORIGINS    = cfg["flip_origins"]
FLAVOUR_TO_LABEL = {int(k): v for k, v in cfg["flavour_to_label"].items()}
CLASS_NAMES     = cfg["class_names"]
COLOURS         = cfg["colours"]

N_FEATS = len(TRACK_FIELDS)

os.makedirs(PLOT_DIR,  exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

_cfg_save_path = os.path.join(PLOT_DIR, "config.json")
with open(_cfg_save_path, "w") as _f:
    json.dump(cfg, _f, indent=4)
print(f"Config saved to {_cfg_save_path}")

# ── data loading ──────────────────────────────────────────────────────
def _cache_key(idx):
    h = hashlib.md5(idx.tobytes()).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"tracks_{h}_nom.npz")


def load_tracks(path, idx):
    """Returns (N, K, F) features, (N, K) validity mask, (N,) labels, (N, K) origins.
    No hard flip — the model's soft flip handles sign flipping internally."""
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
        d0     = f["tracks"]["lifetimeSignedD0"][fidx].astype(np.float32)
        ip2d   = f["tracks"]["lifetimeSignedD0Significance"][fidx].astype(np.float32)
        origin = f["tracks"]["GN2v01_trackOrigin"][fidx].astype(np.int8)
        arrs   = {fld: f["tracks"][fld][fidx].astype(np.float32) for fld in TRACK_FIELDS}

    keep = valid & (np.abs(d0) < 3.5)

    sort_key = ip2d.copy()
    sort_key[~keep] = -np.inf
    order = np.argsort(-sort_key, axis=1)

    feats       = np.stack([arrs[fld] for fld in TRACK_FIELDS], axis=-1)
    topk_idx    = order[:, :TOP_K]
    rows        = np.arange(len(fidx))[:, None]
    topk_feat   = feats[rows, topk_idx]
    topk_valid  = keep[rows, topk_idx]
    topk_feat   = np.where(topk_valid[:, :, None], topk_feat, 0.0).astype(np.float32)
    topk_origin = origin[rows, topk_idx].astype(np.int64)
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



# ── Transformer model ──────────────────────────────────────────────────
class JetTransformer(nn.Module):
    """
    Two-stage transformer with differentiable soft flipping:
      Stage 1 — reads original track features, predicts per-track origin (8 classes).
      Soft flip — for each track, p_flip = sum P(origin=k) for k in flip_origin_idx.
                  flip_scale = -tanh(α*(p_flip - 0.5)): approaches +1 when p_flip < 0.5
                  and -1 when p_flip > 0.5, with sharpness controlled by α.
                  Gradients flow back through p_flip into Stage 1 jointly.
      Stage 2 — reads soft-flipped features, classifies jet flavour.
    Returns: (jet_logits, jet_logits_pre_flip, track_logits_stage1)
      jet_logits          — Stage 2 (post soft-flip), used for CE loss and inference
      jet_logits_pre_flip — Stage 1 (pre soft-flip), used for symmetry loss
      track_logits_stage1 — Stage 1 track origin logits (unflipped), used for origin loss
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

        # Stage 1: original features → per-track origin logits
        self.input_proj = nn.Linear(in_dim, d_model)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.origin_head  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_origins))

        # Stage 2: soft-flipped features → jet classification only
        self.classifier   = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_classes))

        # Registered buffers: which feature dims and which origin classes to soft-flip
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

        # Stage 1 — shared encoder → per-track origin probs
        h1           = self.input_proj(x)
        h1           = torch.cat([self.cls_token.expand(B, -1, -1), h1], dim=1)
        h1           = self.encoder(h1, src_key_padding_mask=src_key_padding_mask)
        track_logits    = self.origin_head(h1[:, 1:])                        # (B, K, N_ORIGINS)
        jet_logits_pre  = self.classifier(h1[:, 0])                          # (B, n_classes) pre-flip

        # Differentiable soft flip — tanh activation centred at p_flip = 0.5
        # scale → +1 when p_flip < 0.5 (keep sign), scale → -1 when p_flip > 0.5 (flip sign)
        origin_probs = torch.softmax(track_logits, dim=-1)                  # (B, K, N_ORIGINS)
        p_flip       = origin_probs[..., self.flip_origin_idx].sum(-1, keepdim=True)  # (B, K, 1)
        flip_scale   = torch.where(
            self.flip_feat_mask,                                            # broadcast over feat dim
            -torch.tanh(self.flip_sharpness * (p_flip - FLIP_THRESHOLD)),             # (B, K, 1)
            torch.ones_like(p_flip),                                        # non-flip features unchanged
        )
        x_soft = x * flip_scale                                             # (B, K, N_FEATS)

        # Stage 2 — shared encoder on soft-flipped features
        h2         = self.input_proj(x_soft)
        h2         = torch.cat([self.cls_token.expand(B, -1, -1), h2], dim=1)
        h2         = self.encoder(h2, src_key_padding_mask=src_key_padding_mask)
        jet_logits = self.classifier(h2[:, 0])                             # CLS → jet flavour

        return jet_logits, jet_logits_pre, track_logits


# ── balanced index sampling ────────────────────────────────────────────
print("Loading data...")
rng = np.random.default_rng(42)

_flavour_cache = os.path.join(CACHE_DIR, "all_flavours.npy")
if os.path.exists(_flavour_cache):
    all_flavours = np.load(_flavour_cache)
else:
    with h5py.File(TRAIN_FILE, "r") as f:
        all_flavours = f["jets"]["HadronConeExclTruthLabelID"][:]
    np.save(_flavour_cache, all_flavours)

valid_mask = np.isin(all_flavours, list(FLAVOUR_TO_LABEL.keys()))
valid_idx  = rng.permutation(np.where(valid_mask)[0])

test_idx    = np.sort(valid_idx[-N_TEST:])
pool_idx    = valid_idx[:-N_TEST]
pool_labels = np.array([FLAVOUR_TO_LABEL[v] for v in all_flavours[pool_idx]])

n_per_class = N_TRAIN // 3
train_idx   = np.sort(np.concatenate([
    rng.choice(pool_idx[pool_labels == cls],
               size=min(n_per_class, (pool_labels == cls).sum()),
               replace=False)
    for cls in range(3)
]))

X_train, mask_train, y_train, origins_train = load_tracks(TRAIN_FILE, train_idx)
X_test,  mask_test,  y_test,  origins_test  = load_tracks(TRAIN_FILE, test_idx)

print(f"Train — b:{(y_train==0).sum():,}  c:{(y_train==1).sum():,}  light:{(y_train==2).sum():,}")
print(f"Test  — b:{(y_test==0).sum():,}  c:{(y_test==1).sum():,}  light:{(y_test==2).sum():,}")

_pin = DEVICE == "cuda"
_pw  = NUM_WORKERS > 0
train_loader = DataLoader(
    JetDataset(X_train, mask_train, y_train, origins_train),
    batch_size=BATCH_SIZE, shuffle=True, pin_memory=_pin,
    num_workers=NUM_WORKERS, persistent_workers=_pw)
val_loader = DataLoader(
    JetDataset(X_test, mask_test, y_test, origins_test),
    batch_size=BATCH_SIZE, pin_memory=_pin,
    num_workers=NUM_WORKERS, persistent_workers=_pw)

# ── input variable plot ────────────────────────────────────────────────
tracks_flat = X_train.reshape(-1, N_FEATS)
labels_rep  = np.repeat(y_train, TOP_K)
nonzero     = mask_train.ravel()

n_cols = min(N_FEATS, 4)
n_rows = (N_FEATS + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
axes = np.array(axes).ravel()
fig.suptitle("Input variables by jet flavour (training sample)", fontweight="bold")
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
plt.savefig(PLOT_DIR + "input_variables.png", dpi=150, bbox_inches="tight")
print("Saved input_variables.png")

# ── model ─────────────────────────────────────────────────────────────
_flip_feat_idx   = [TRACK_FIELDS.index(f) for f in FLIP_FIELDS]
_flip_origin_idx = list(range(N_ORIGINS)) if FLIP_ORIGINS is None else list(FLIP_ORIGINS)

model            = JetTransformer(N_FEATS, D_MODEL, N_HEADS, N_LAYERS, D_FFN, DROPOUT,
                                  n_classes=3, n_origins=N_ORIGINS,
                                  flip_feat_indices=_flip_feat_idx,
                                  flip_origin_indices=_flip_origin_idx,
                                  flip_sharpness=FLIP_SHARPNESS).to(DEVICE)
optimiser        = torch.optim.Adam(model.parameters(), lr=LR)
criterion        = nn.CrossEntropyLoss()
criterion_origin = nn.CrossEntropyLoss(ignore_index=-1)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {n_params:,}")
print(f"Device: {DEVICE}  |  Train: {len(y_train):,}  |  Test: {len(y_test):,}\n")

# ── training loop ─────────────────────────────────────────────────────
history = {"train_loss": [], "train_ce_loss": [], "train_sym_loss": [], "train_origin_loss": [], "val_loss": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, total_ce, total_sym, total_orig = 0.0, 0.0, 0.0, 0.0

    for X_b, mask_b, y_b, origins_b in train_loader:
        X_b, mask_b, y_b = X_b.to(DEVICE), mask_b.to(DEVICE), y_b.to(DEVICE)
        origins_b = origins_b.to(DEVICE)
        optimiser.zero_grad()

        logits, logits_pre, track_logits_1 = model(X_b, mask_b)

        ce_loss = criterion(logits_pre, y_b)

        # Symmetry loss: post-flip vs pre-flip jet probs
        p_post = torch.softmax(logits,     dim=1)
        p_pre  = torch.softmax(logits_pre, dim=1)
        light_mask = (y_b == 2)
        b_mask     = (y_b == 0)
        if light_mask.any():
            p_pre_light  = p_pre[light_mask]
            p_post_light = p_post[light_mask]
            # b-score for light jets from the pre-flip stage
            b_score_light = p_pre_light[:, 0]
            top10_thresh  = torch.quantile(b_score_light, 0.90)
            top10_mask    = b_score_light >= top10_thresh
            sym_loss_light = F.mse_loss(p_post_light[top10_mask], p_pre_light[top10_mask]) if top10_mask.any() else logits.new_tensor(0.0)
        else:
            sym_loss_light = logits.new_tensor(0.0)
        sym_loss_b     = F.mse_loss(p_post[b_mask],     p_pre[b_mask])     if b_mask.any()     else logits.new_tensor(0.0)
        sym_loss = sym_loss_light - B_RATIO * sym_loss_b

        # Track origin classification loss (Stage 1, unflipped, ignore padding=-1)
        origin_loss = criterion_origin(
            track_logits_1.reshape(-1, N_ORIGINS),
            origins_b.reshape(-1),
        )

        loss = ce_loss + LAMBDA_SYM * sym_loss + LAMBDA_ORIG * origin_loss

        loss.backward()
        optimiser.step()
        total_loss += loss.item()        * len(y_b)
        total_ce   += ce_loss.item()     * len(y_b)
        total_sym  += sym_loss.item()    * len(y_b)
        total_orig += origin_loss.item() * len(y_b)

    train_loss        = total_loss / len(y_train)
    train_ce_loss     = total_ce   / len(y_train)
    train_sym_loss    = total_sym  / len(y_train)
    train_origin_loss = total_orig / len(y_train)

    model.eval()
    val_loss, correct = 0.0, 0
    all_preds, all_true, all_probs = [], [], []
    all_origin_preds, all_origin_true = [], []
    with torch.no_grad():
        for X_b, mask_b, y_b, origins_b in val_loader:
            X_b, mask_b, y_b = X_b.to(DEVICE), mask_b.to(DEVICE), y_b.to(DEVICE)
            logits, _, track_logits_1 = model(X_b, mask_b)
            val_loss += criterion(logits, y_b).item() * len(y_b)
            preds = logits.argmax(dim=1)
            correct += (preds == y_b).sum().item()
            all_preds.append(preds.cpu())
            all_true.append(y_b.cpu())
            all_probs.append(torch.softmax(logits, dim=1).cpu())
            all_origin_preds.append(track_logits_1.argmax(dim=-1).cpu())  # (B, K) — Stage 1
            all_origin_true.append(origins_b)
    val_loss /= len(y_test)
    val_acc   = correct / len(y_test)

    history["train_loss"].append(train_loss)
    history["train_ce_loss"].append(train_ce_loss)
    history["train_sym_loss"].append(train_sym_loss)
    history["train_origin_loss"].append(train_origin_loss)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    print(f"Epoch {epoch:02d}/{EPOCHS}  "
          f"loss={train_loss:.4f}  ce={train_ce_loss:.4f}  "
          f"sym={train_sym_loss:.4f}  orig={train_origin_loss:.4f}  "
          f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

# ── save model ────────────────────────────────────────────────────────
torch.save(model.state_dict(), MODEL_NAME)
print(f"Saved {MODEL_NAME}")

# ── final evaluation ──────────────────────────────────────────────────
all_preds = torch.cat(all_preds).numpy()
all_true  = torch.cat(all_true).numpy()
all_probs = torch.cat(all_probs).numpy()

# flatten track origin arrays and strip padding (true origin == -1)
_op = torch.cat(all_origin_preds).numpy().ravel()
_ot = torch.cat(all_origin_true).numpy().ravel()
valid_tracks   = _ot >= 0
origin_preds   = _op[valid_tracks]
origin_true    = _ot[valid_tracks]

print("\nClassification report:")
print(classification_report(all_true, all_preds, target_names=CLASS_NAMES))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(all_true, all_preds))

# ── plots ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Transformer jet classifier — training summary", fontweight="bold")
ep = range(1, EPOCHS + 1)
axes[0].plot(ep, history["train_loss"],        label="train (total)")
axes[0].plot(ep, history["train_ce_loss"],     label="train CE",              linestyle="--")
axes[0].plot(ep, history["train_sym_loss"],    label=f"train sym (×{LAMBDA_SYM})",  linestyle=":")
axes[0].plot(ep, history["train_origin_loss"], label=f"train orig (×{LAMBDA_ORIG})", linestyle="-.")
axes[0].plot(ep, history["val_loss"],          label="val")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend(fontsize=7)

axes[1].plot(ep, history["val_acc"])
axes[1].set_title("Val jet accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1)

cm = confusion_matrix(all_true, all_preds, normalize="true")
im = axes[2].imshow(cm, cmap="Blues", vmin=0, vmax=1)
axes[2].set_xticks([0,1,2]); axes[2].set_yticks([0,1,2])
axes[2].set_xticklabels(["b","c","light"]); axes[2].set_yticklabels(["b","c","light"])
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")
axes[2].set_title("Confusion matrix (normalised)")
for i in range(3):
    for j in range(3):
        axes[2].text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                     color="white" if cm[i,j] > 0.5 else "black")
plt.colorbar(im, ax=axes[2])
plt.tight_layout()
plt.savefig(PLOT_DIR + "transformer_results.png", dpi=150, bbox_inches="tight")
print("Saved transformer_results.png")

# ── output probabilities ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Transformer output probabilities by true flavour (test sample)", fontweight="bold")
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    ax = axes[cls_idx]
    for true_idx, true_name in enumerate(CLASS_NAMES):
        ax.hist(all_probs[all_true == true_idx, cls_idx], bins=50, range=(0, 1),
                histtype="step", label=true_name, color=COLOURS[true_name],
                linewidth=1.5, density=True)
    ax.set_title(f"P({cls_name})"); ax.set_xlabel("Probability")
    ax.set_ylabel("Density"); ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(PLOT_DIR + "transformer_output_probs.png", dpi=150, bbox_inches="tight")
print("Saved transformer_output_probs.png")

# ── discriminant ──────────────────────────────────────────────────────
pb, pc, pu = all_probs[:, 0], all_probs[:, 1], all_probs[:, 2]
disc   = np.log(pb / (0.2 * pc + 0.8 * pu + 1e-10))
finite = np.isfinite(disc)
clip   = np.percentile(np.abs(disc[finite]), 99)

fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle(r"$\log(p_b\,/\,(0.2\,p_c + 0.8\,p_u))$ — test sample", fontweight="bold")
for true_idx, true_name in enumerate(CLASS_NAMES):
    mask = (all_true == true_idx) & finite
    ax.hist(disc[mask], bins=80, range=(-clip, clip), histtype="step",
            label=true_name, color=COLOURS[true_name], linewidth=1.5, density=True)
ax.set_xlabel(r"$\log(p_b\,/\,(0.2\,p_c + 0.8\,p_u))$")
ax.set_ylabel("Density"); ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR + "transformer_discriminant.png", dpi=150, bbox_inches="tight")
print("Saved transformer_discriminant.png")

# ── ROC curves ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(r"ROC curves — $\log(p_b\,/\,(0.2\,p_c + 0.8\,p_u))$", fontweight="bold")
for ax, (bkg_idx, bkg_name) in zip(axes, [(1, "c-jet"), (2, "light-jet")]):
    mask   = (all_true == 0) | (all_true == bkg_idx)
    labels = (all_true[mask] == 0).astype(int)
    score  = disc[mask]
    fin    = np.isfinite(score)
    fpr, tpr, _ = roc_curve(labels[fin], score[fin])
    ax.plot(tpr, fpr, color="#1f77b4", linewidth=1.5, label=f"AUC={auc(fpr,tpr):.3f}")
    ax.set_xlabel("b-jet efficiency (TPR)")
    ax.set_ylabel(f"{bkg_name} rate (FPR)")
    ax.set_title(f"b vs {bkg_name}"); ax.set_yscale("log")
    ax.legend(); ax.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(PLOT_DIR + "transformer_roc.png", dpi=150, bbox_inches="tight")
print("Saved transformer_roc.png")

# ── track origin confusion matrix ─────────────────────────────────────
ORIGIN_NAMES = ["Pileup", "Fake", "Primary", "From b", "From b→c", "From c", "From τ", "Other sec."]
present      = sorted(np.unique(origin_true))
labels_pres  = [ORIGIN_NAMES[i] for i in present]

cm_orig = confusion_matrix(origin_true, origin_preds, labels=present, normalize="true")
fig, ax = plt.subplots(figsize=(9, 7))
fig.suptitle("Track origin confusion matrix (normalised, Stage 2, test sample)", fontweight="bold")
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
plt.savefig(PLOT_DIR + "transformer_origin_confusion.png", dpi=150, bbox_inches="tight")
print("Saved transformer_origin_confusion.png")
