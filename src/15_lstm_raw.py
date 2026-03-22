"""
Phase 5 - Step 2: LSTM on Raw OHLCV Sequences
==============================================
Hypothesis: raw price sequences contain momentum patterns that engineered
features destroy. Train a 2-layer LSTM on 60-day rolling windows of
OHLCV data, normalised with rolling 252-day z-scores, to predict dir_1w.

Input  : 5 raw features (open, high, low, close, volume)
         Each normalised per-day as z = (x - roll_mean_252) / roll_std_252
Seq len: 60 trading days (3 months)
Target : dir_1w (Bear=-1, Sideways=0, Bull=+1)

Architecture vs Step 1 changes:
  input_size  : 5  (was 57)
  hidden_size : 256 (was 128)
  dropout     : 0.4 (was 0.3)
  weight_decay: 1e-4 in Adam (was 0.0)
  seq_len     : 60  (was 100)

Evaluation:
  Walk-forward CV (5 folds) on 1995-2023  -> CV Macro F1
  Holdout 2024-01-01 to 2025-03-19        -> Holdout Macro F1
  Compare both against XGBoost champion (F1=0.375)

Outputs:
  models/lstm_raw_ohlcv.pt
  data/processed/aapl_predictions_lstm_raw.parquet
  docs/lstm_raw_loss_curves.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, classification_report, recall_score
from sklearn.utils.class_weight import compute_class_weight

from config import paths as P, settings as S

warnings.filterwarnings("ignore")
torch.manual_seed(42)
np.random.seed(42)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET        = "dir_1w"
SEQ_LEN       = S.SEQ_LEN_RAW          # 60 days
ROLL_WIN      = S.ROLLING_ZSCORE_WIN   # 252 days
HOLDOUT_START = S.HOLDOUT_START
N_SPLITS      = S.N_SPLITS

FEATURES      = S.OHLCV_FEATURES       # ["open","high","low","close","volume"]
ARCH          = S.LSTM_RAW_ARCH
TRAIN         = S.LSTM_RAW_TRAIN

LABEL_ENCODE  = S.LABEL_ENCODE
LABEL_DECODE  = S.LABEL_DECODE
CLASS_LABELS  = S.CLASS_LABELS

P3            = S.PHASE3_CHAMPION


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2,
                 num_classes=3, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0.0,
            batch_first = True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out    = self.dropout(out[:, -1, :])
        return self.fc(out)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SequenceDataset(Dataset):
    def __init__(self, X_seq: np.ndarray, y_seq: np.ndarray):
        self.X = torch.FloatTensor(X_seq)
        self.y = torch.LongTensor(y_seq)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def rolling_zscore(df: pd.DataFrame, cols: list, window: int,
                   min_periods: int = 50, clip: float = 5.0) -> pd.DataFrame:
    """
    Replace each column with its rolling z-score using strictly past data.
    z_t = (x_t - mean(x_{t-window+1:t})) / std(x_{t-window+1:t})
    Clips extreme values at ±clip to suppress outlier spikes (volume etc.)
    Returns a copy with the normalised columns.
    """
    out = df.copy()
    for col in cols:
        roll   = df[col].rolling(window, min_periods=min_periods)
        mu     = roll.mean()
        sigma  = roll.std().replace(0, np.nan)
        z      = (df[col] - mu) / sigma
        out[col] = z.clip(-clip, clip)
    return out


def build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Build (X_seq, y_seq, end_indices) where X_seq[i] = X[i-seq_len+1 : i+1]
    and y_seq[i] = y[i].  Returns only positions where i >= seq_len-1.
    """
    n           = len(X)
    end_indices = np.arange(seq_len - 1, n)
    X_seq       = np.stack([X[i - seq_len + 1: i + 1] for i in end_indices])
    y_seq       = y[end_indices]
    return X_seq, y_seq, end_indices


def make_loader(X_seq, y_seq, batch_size, shuffle):
    ds = SequenceDataset(X_seq, y_seq)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=torch.cuda.is_available(),
    )


def section(title):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


# ══════════════════════════════════════════════════════════════════════════════
# Training loop
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(model, train_loader, val_loader, device, class_weights):
    """Train with early stopping + ReduceLROnPlateau + AMP."""
    crit      = nn.CrossEntropyLoss(weight=class_weights.to(device))
    opt       = torch.optim.Adam(
        model.parameters(),
        lr           = TRAIN["lr"],
        weight_decay = TRAIN["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=TRAIN["lr_factor"],
        patience=TRAIN["patience_lr"], min_lr=TRAIN["lr_min"],
    )
    use_amp    = device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_loss = float("inf")
    best_state    = None
    best_epoch    = 1
    patience_ctr  = 0
    train_losses  = []
    val_losses    = []

    for epoch in range(1, TRAIN["max_epochs"] + 1):

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(X_b)
                loss   = crit(logits, y_b)
            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
            scaler_amp.step(opt)
            scaler_amp.update()
            epoch_loss += loss.item() * len(y_b)

        train_loss = epoch_loss / len(train_loader.dataset)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        val_loss  = 0.0
        all_preds = []
        all_true  = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(X_b)
                    loss   = crit(logits, y_b)
                val_loss  += loss.item() * len(y_b)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_true.extend(y_b.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        val_f1    = f1_score(all_true, all_preds, average="macro", zero_division=0)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        # ── Early stopping bookkeeping ─────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch    = epoch
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr  = 0
        else:
            patience_ctr += 1

        # ── Logging ────────────────────────────────────────────────────────
        if epoch % 10 == 0 or epoch <= 3 or patience_ctr >= TRAIN["patience_es"]:
            lr_now = opt.param_groups[0]["lr"]
            print(f"    Epoch {epoch:3d} | train={train_loss:.4f}  val={val_loss:.4f}"
                  f"  F1={val_f1:.3f}  lr={lr_now:.2e}  patience={patience_ctr}/{TRAIN['patience_es']}")

        if patience_ctr >= TRAIN["patience_es"]:
            print(f"    Early stopping at epoch {epoch}  (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    return model, train_losses, val_losses, best_epoch


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── GPU check ─────────────────────────────────────────────────────────────
    section("GPU / Device")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        device   = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU    : {gpu_name}")
        print(f"  CUDA   : {torch.version.cuda}  |  PyTorch {torch.__version__}")
        print(f"  VRAM   : {vram_gb:.1f} GB")
    else:
        device = torch.device("cpu")
        print("  No GPU found — running on CPU")
    print(f"  Device : {device}")

    # ── Load raw OHLCV ────────────────────────────────────────────────────────
    section("Loading & Normalising Data")

    raw = pd.read_parquet(P.DATA_RAW)
    raw.index = pd.to_datetime(raw.index).normalize()
    raw       = raw.sort_index()[FEATURES]
    print(f"  Raw OHLCV rows : {len(raw):,}  "
          f"({raw.index.min().date()} to {raw.index.max().date()})")

    # Rolling 252-day z-score normalisation (strictly backward-looking)
    raw_z = rolling_zscore(raw, FEATURES, window=ROLL_WIN, min_periods=50)
    before_drop = len(raw_z)
    raw_z = raw_z.dropna()
    print(f"  After z-score NaN drop : {len(raw_z):,} rows  "
          f"(dropped {before_drop - len(raw_z)} warm-up days)")
    print(f"  Feature stats after normalisation:")
    for col in FEATURES:
        s = raw_z[col]
        print(f"    {col:<8}  mean={s.mean():+.3f}  std={s.std():.3f}"
              f"  min={s.min():.2f}  max={s.max():.2f}")

    # ── Load labels ───────────────────────────────────────────────────────────
    labels = pd.read_parquet(P.DATA_WITH_EVENTS)[[TARGET]]
    labels.index = pd.to_datetime(labels.index).normalize()
    labels = labels.dropna(subset=[TARGET])

    # Join on common dates
    df = raw_z.join(labels, how="inner")
    df = df.dropna(subset=[TARGET])
    df["label_enc"] = df[TARGET].map(LABEL_ENCODE)

    print(f"\n  After joining labels : {len(df):,} rows  "
          f"({df.index.min().date()} to {df.index.max().date()})")

    # ── Train/val vs holdout ───────────────────────────────────────────────────
    train_val_df = df[df.index <  HOLDOUT_START].copy()
    holdout_df   = df[df.index >= HOLDOUT_START].copy()

    print(f"  Train/val : {len(train_val_df):,} rows  "
          f"({train_val_df.index.min().date()} to {train_val_df.index.max().date()})")
    print(f"  Holdout   : {len(holdout_df):,} rows  "
          f"({holdout_df.index.min().date()} to {holdout_df.index.max().date()})")

    dist = train_val_df[TARGET].value_counts().sort_index()
    print(f"  Train/val label dist: Bear={dist.get(-1,0):,} "
          f"Side={dist.get(0,0):,}  Bull={dist.get(1,0):,}")

    X_tv    = train_val_df[FEATURES].values.astype(np.float32)
    y_tv    = train_val_df["label_enc"].values.astype(np.int64)

    X_ho    = holdout_df[FEATURES].values.astype(np.float32)
    y_ho    = holdout_df["label_enc"].values.astype(np.int64)

    input_size = len(FEATURES)  # 5

    # ══════════════════════════════════════════════════════════════════════════
    # Walk-forward CV
    # ══════════════════════════════════════════════════════════════════════════
    section("Walk-Forward Cross-Validation")
    print(f"  Sequence length : {SEQ_LEN} days  |  Input features : {input_size}")
    print(f"  Hidden size     : {ARCH['hidden_size']}  |  Dropout : {ARCH['dropout']}")
    print(f"  Weight decay    : {TRAIN['weight_decay']}")

    tscv         = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_f1s     = []
    fold_losses  = []
    fold_epochs  = []
    all_cv_preds = np.full(len(train_val_df), -999, dtype=np.int64)

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_tv)):
        section(f"Fold {fold_idx + 1} / {N_SPLITS}")
        tr_dates = train_val_df.index[train_idx]
        vl_dates = train_val_df.index[val_idx]
        print(f"  Train: {len(train_idx):,} rows  "
              f"({tr_dates.min().date()} to {tr_dates.max().date()})")
        print(f"  Val  : {len(val_idx):,} rows  "
              f"({vl_dates.min().date()} to {vl_dates.max().date()})")

        # Rolling z-score already applied — no per-fold scaler needed.
        # Build sequences over train+val window (val sequences use train as context).
        max_pos      = val_idx[-1] + 1
        X_window     = X_tv[:max_pos]
        y_window     = y_tv[:max_pos]

        X_seq, y_seq, end_idx = build_sequences(X_window, y_window, SEQ_LEN)

        train_end  = train_idx[-1]
        is_val_seq = end_idx > train_end

        X_tr_seq   = X_seq[~is_val_seq];  y_tr_seq = y_seq[~is_val_seq]
        X_vl_seq   = X_seq[is_val_seq];   y_vl_seq = y_seq[is_val_seq]
        vl_end_idx = end_idx[is_val_seq]

        print(f"  Train sequences : {len(X_tr_seq):,}  |  Val sequences : {len(X_vl_seq):,}")

        if len(X_vl_seq) == 0:
            print("  Skipping fold — not enough val sequences")
            continue

        # Class weights on training sequences
        cw            = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y_tr_seq)
        class_weights = torch.FloatTensor(cw)
        print(f"  Class weights  : Bear={cw[0]:.3f}  Side={cw[1]:.3f}  Bull={cw[2]:.3f}")

        train_loader = make_loader(X_tr_seq, y_tr_seq, TRAIN["batch_size"], shuffle=True)
        val_loader   = make_loader(X_vl_seq, y_vl_seq, TRAIN["batch_size"], shuffle=False)

        model = StockLSTM(
            input_size  = input_size,
            hidden_size = ARCH["hidden_size"],
            num_layers  = ARCH["num_layers"],
            num_classes = 3,
            dropout     = ARCH["dropout"],
        ).to(device)

        model, tr_losses, vl_losses, best_ep = train_fold(
            model, train_loader, val_loader, device, class_weights,
        )
        fold_losses.append((tr_losses, vl_losses))
        fold_epochs.append(best_ep)

        # Val predictions
        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_b, _ in val_loader:
                val_preds.extend(
                    model(X_b.to(device)).argmax(dim=1).cpu().numpy()
                )
        val_preds = np.array(val_preds)

        fold_f1 = f1_score(y_vl_seq, val_preds, average="macro", zero_division=0)
        fold_f1s.append(fold_f1)

        for pos, pred in zip(vl_end_idx, val_preds):
            all_cv_preds[pos] = pred

        print(f"\n  Fold {fold_idx + 1} Val Macro F1 = {fold_f1:.4f}")
        print(classification_report(
            y_vl_seq, val_preds, target_names=CLASS_LABELS, zero_division=0,
        ))

    cv_macro_f1 = float(np.mean(fold_f1s))
    avg_best_ep = int(np.round(np.mean(fold_epochs)))

    print(f"\n{'=' * 62}")
    print(f"  CV Macro F1 (mean {N_SPLITS} folds)  : {cv_macro_f1:.4f}")
    print(f"  XGB P3 champion            : {P3['f1']:.4f}")
    delta = cv_macro_f1 - P3["f1"]
    sign  = "+" if delta >= 0 else ""
    print(f"  Delta                      : {sign}{delta:.4f}")
    print(f"  Avg best epoch across folds: {avg_best_ep}")
    print(f"  Fold F1s                   : {[round(f, 3) for f in fold_f1s]}")

    # ── Save loss curves ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, N_SPLITS, figsize=(4 * N_SPLITS, 4), sharey=False)
    for i, (tl, vl) in enumerate(fold_losses):
        ax = axes[i] if N_SPLITS > 1 else axes
        ax.plot(tl, label="train", color="steelblue")
        ax.plot(vl, label="val",   color="tomato")
        ax.axvline(fold_epochs[i] - 1, color="green", linestyle="--",
                   linewidth=0.8, label=f"best={fold_epochs[i]}")
        ax.set_title(f"Fold {i+1}  F1={fold_f1s[i]:.3f}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
    plt.suptitle("LSTM (Raw OHLCV) Train vs Val Loss — Walk-Forward CV", fontsize=11)
    plt.tight_layout()
    P.LOSS_CURVES_LSTM_RAW.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(P.LOSS_CURVES_LSTM_RAW, dpi=120)
    plt.close()
    print(f"\n  Loss curves saved -> {P.LOSS_CURVES_LSTM_RAW}")

    # ══════════════════════════════════════════════════════════════════════════
    # Final model — retrain on full train/val
    # ══════════════════════════════════════════════════════════════════════════
    section("Final Model — Full Train/Val Retrain")
    print(f"  Training for up to {TRAIN['max_epochs']} epochs  "
          f"(avg best from CV: {avg_best_ep})")

    X_seq_full, y_seq_full, _ = build_sequences(X_tv, y_tv, SEQ_LEN)

    n_total  = len(X_seq_full)
    n_val_es = max(200, int(n_total * 0.15))
    X_f_tr   = X_seq_full[:-n_val_es];  y_f_tr = y_seq_full[:-n_val_es]
    X_f_vl   = X_seq_full[-n_val_es:];  y_f_vl = y_seq_full[-n_val_es:]

    cw_final  = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y_f_tr)
    cw_tensor = torch.FloatTensor(cw_final)
    print(f"  Total sequences : {n_total:,}  |  ES val : {n_val_es:,}")
    print(f"  Class weights   : Bear={cw_final[0]:.3f}  Side={cw_final[1]:.3f}  Bull={cw_final[2]:.3f}")

    f_train_loader = make_loader(X_f_tr, y_f_tr, TRAIN["batch_size"], shuffle=True)
    f_val_loader   = make_loader(X_f_vl, y_f_vl, TRAIN["batch_size"], shuffle=False)

    final_model = StockLSTM(
        input_size  = input_size,
        hidden_size = ARCH["hidden_size"],
        num_layers  = ARCH["num_layers"],
        num_classes = 3,
        dropout     = ARCH["dropout"],
    ).to(device)

    final_model, _, _, final_best_ep = train_fold(
        final_model, f_train_loader, f_val_loader, device, cw_tensor,
    )
    print(f"  Final model best epoch: {final_best_ep}")

    # ══════════════════════════════════════════════════════════════════════════
    # Holdout evaluation
    # ══════════════════════════════════════════════════════════════════════════
    section("Holdout Evaluation (2024-01-01 to end)")

    # Holdout sequences use train_val as context — combine arrays
    X_combined    = np.vstack([X_tv, X_ho])
    y_combined    = np.concatenate([y_tv, y_ho])

    X_seq_comb, y_seq_comb, end_idx_comb = build_sequences(
        X_combined, y_combined, SEQ_LEN
    )

    ho_start   = len(X_tv)
    is_holdout = end_idx_comb >= ho_start
    X_ho_seq   = X_seq_comb[is_holdout]
    y_ho_seq   = y_seq_comb[is_holdout]
    ho_end_idx = end_idx_comb[is_holdout] - ho_start

    print(f"  Holdout sequences : {len(X_ho_seq):,}")

    final_model.eval()
    ho_preds  = []
    ho_loader = DataLoader(
        SequenceDataset(X_ho_seq, y_ho_seq),
        batch_size=256, shuffle=False, num_workers=0,
    )
    with torch.no_grad():
        for X_b, _ in ho_loader:
            ho_preds.extend(
                final_model(X_b.to(device)).argmax(dim=1).cpu().numpy()
            )
    ho_preds = np.array(ho_preds)

    ho_f1     = f1_score(y_ho_seq, ho_preds, average="macro", zero_division=0)
    ho_acc    = float((y_ho_seq == ho_preds).mean())
    ho_recall = recall_score(y_ho_seq, ho_preds, average=None,
                             labels=[0,1,2], zero_division=0)

    print(f"\n  Holdout Accuracy  : {ho_acc * 100:.2f}%")
    print(f"  Holdout Macro F1  : {ho_f1:.4f}")
    print(classification_report(
        y_ho_seq, ho_preds, target_names=CLASS_LABELS, zero_division=0,
    ))

    # ── Save model ────────────────────────────────────────────────────────────
    P.MODEL_LSTM_RAW.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state" : final_model.state_dict(),
        "features"    : FEATURES,
        "seq_len"     : SEQ_LEN,
        "roll_win"    : ROLL_WIN,
        "input_size"  : input_size,
        "lstm_arch"   : ARCH,
        "cv_f1"       : cv_macro_f1,
        "holdout_f1"  : ho_f1,
        "holdout_acc" : ho_acc,
    }, P.MODEL_LSTM_RAW)
    print(f"\n  Model saved -> {P.MODEL_LSTM_RAW}")

    # ── Save predictions ──────────────────────────────────────────────────────
    cv_mask  = all_cv_preds != -999
    cv_out   = train_val_df[cv_mask][["dir_1w"]].copy()
    cv_out["pred_enc"] = all_cv_preds[cv_mask]
    cv_out["pred"]     = cv_out["pred_enc"].map(LABEL_DECODE)
    cv_out["actual"]   = cv_out["dir_1w"]
    cv_out["split"]    = "cv"

    ho_out             = holdout_df.iloc[ho_end_idx][["dir_1w"]].copy()
    ho_out["pred_enc"] = ho_preds
    ho_out["pred"]     = ho_out["pred_enc"].map(LABEL_DECODE)
    ho_out["actual"]   = ho_out["dir_1w"]
    ho_out["split"]    = "holdout"

    pred_df = pd.concat([cv_out[["actual", "pred", "split"]],
                         ho_out[["actual", "pred", "split"]]])
    P.PRED_LSTM_RAW.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(P.PRED_LSTM_RAW)
    print(f"  Predictions saved -> {P.PRED_LSTM_RAW}")

    # ══════════════════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════════════════
    section("PHASE 5 STEP 2 — RESULTS SUMMARY")

    p3r = P3["recall"]
    print(f"""
  Model                             Acc       F1     Bear     Side     Bull
  XGB P3 champion (57 feat)      38.30%   {P3['f1']:.3f}   {p3r['Bear']*100:.1f}%   {p3r['Sideways']*100:.1f}%   {p3r['Bull']*100:.1f}%
  LSTM engineered (Step 1, CV)      --    0.297     --       --       --
  LSTM raw OHLCV  (Step 2, CV)      --    {cv_macro_f1:.3f}     --       --       --
  LSTM raw OHLCV  (Holdout)      {ho_acc*100:.2f}%   {ho_f1:.3f}   {ho_recall[0]*100:.1f}%   {ho_recall[1]*100:.1f}%   {ho_recall[2]*100:.1f}%
    """)

    if cv_macro_f1 > P3["f1"]:
        verdict = f"LSTM raw OHLCV BEATS XGB champion on CV F1 (+{cv_macro_f1 - P3['f1']:.3f})"
    elif cv_macro_f1 > 0.297:
        verdict = (f"Raw OHLCV improves over engineered LSTM "
                   f"(+{cv_macro_f1 - 0.297:.3f}), XGB champion still leads")
    else:
        verdict = f"No improvement over engineered LSTM — XGB champion remains best"

    print(f"  Verdict: {verdict}")
    print(f"""
  Outputs:
    Model       -> {P.MODEL_LSTM_RAW}
    Predictions -> {P.PRED_LSTM_RAW}
    Loss curves -> {P.LOSS_CURVES_LSTM_RAW}

Phase 5 Step 2 complete.
""")
