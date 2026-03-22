"""
Phase 4 - Step 2: Stacking Ensemble  (XGBoost + LightGBM)
==========================================================
Stacks the Phase 3 XGBoost champion and the Phase 4 LightGBM model using
their out-of-sample probability outputs as meta-features.

Architecture
------------
Level 0  (base models, both retrained from scratch in each fold):
  - XGBoost  : same params as Phase 3 champion (xgb_best_interactions)
  - LightGBM : same params as Phase 4 Step 1 (lgbm_dir_1w)

Level 1  (meta-learner):
  - 12 meta-features: xgb_prob_bear/side/bull + lgbm_prob_bear/side/bull
  - LogisticRegressionCV  (cross-validates C ∈ [0.01..100], multinomial)

Zero-leakage walk-forward stacking protocol
-------------------------------------------
Phase 1 — 5 expanding folds, both base models trained on fold.train,
          OOS probas generated on fold.test.  All 5 folds produce
          genuine OOS probas (never seen during that fold's training).

Phase 2 — Meta-learner is walk-forward too:
          fold k test is evaluated using a meta-learner trained only on
          OOS meta-features from folds 1..(k-1).  This means:
            • Fold 1  — no prior meta data; excluded from OOS evaluation
            • Fold 2  — meta trained on fold 1 meta-features (1,225 rows)
            • Fold 3  — meta trained on folds 1+2  (2,450 rows)
            • Fold 4  — meta trained on folds 1+2+3 (3,675 rows)
            • Fold 5  — meta trained on folds 1+2+3+4 (4,900 rows)
          Combined OOS evaluation covers folds 2-5  (4,900 rows / ~80%).

Final model — base models retrained on full dataset, meta-learner trained
              on all 5 folds of accumulated OOS meta-features (6,125 rows).

Phase 3 champion baseline (XGBoost, dir_1w, 57 feat):
  OOS Accuracy : 38.30%   Naive: 37.50%
  Macro F1     : 0.375
  Per-class    : Bear=30.6%  Sideways=39.9%  Bull=42.0%

Outputs:
  models/ensemble_v1.pkl          (saved if ensemble beats XGBoost)
  data/processed/aapl_predictions_ensemble.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pickle

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from config import paths as P, settings as S

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_FILE        = P.DATA_WITH_EVENTS
FEAT_FILE        = P.FEATURE_LIST
ENSEMBLE_FILE    = P.ROOT / "models" / "ensemble_v1.pkl"
PRED_FILE        = P.PRED_ENSEMBLE

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET       = "dir_1w"
N_SPLITS     = S.N_SPLITS
LABEL_ENCODE = S.LABEL_ENCODE
LABEL_DECODE = S.LABEL_DECODE
CLASS_NAMES  = S.CLASS_NAMES
CLASS_LABELS = S.CLASS_LABELS
CLASSES      = S.CLASSES
EVENT_FEATURES       = S.EVENT_FEATURES
INTERACTION_FEATURES = S.INTERACTION_FEATURES

# XGBoost — identical to Phase 3 champion
XGB_PARAMS = S.XGB_PARAMS

# LightGBM — identical to Phase 4 Step 1
LGBM_PARAMS = {
    "n_estimators"     : 300,
    "num_leaves"       : 31,
    "max_depth"        : 4,
    "learning_rate"    : 0.05,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "min_child_samples": 20,
    "objective"        : "multiclass",
    "num_class"        : 3,
    "metric"           : "multi_logloss",
    "random_state"     : 42,
    "n_jobs"           : -1,
    "verbosity"        : -1,
}

# Meta-learner: LogisticRegressionCV (cross-validates regularisation C)
# class_weight="balanced" prevents the meta-learner collapsing to the
# majority class when strong L2 regularisation is chosen by CV.
META_PARAMS = {
    "Cs"          : [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "cv"          : 3,          # 3-fold inner CV to choose C
    "class_weight": "balanced",
    "solver"      : "lbfgs",
    "max_iter"    : 1000,
    "random_state": 42,
    "n_jobs"      : -1,
}

# Phase benchmarks
P3  = S.PHASE3_CHAMPION     # XGBoost F1=0.375
P4L = {"acc": 0.3607, "f1": 0.3532,
       "recall": {"Bear": 0.2956, "Sideways": 0.3396, "Bull": 0.4279}}


# ── Helpers ────────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


def naive_baseline_accuracy(y_true):
    counts = pd.Series(y_true).value_counts()
    return counts.idxmax(), counts.max() / len(y_true)


def zscore_global(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std() + 1e-9)


def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    days_remaining = (90 - d["days_to_next_earnings"]).clip(lower=0)
    d["earnings_proximity_surprise"] = days_remaining * d["last_eps_surprise_pct"].abs()
    z_rate = zscore_global(d["fed_rate_change_3m"])
    z_cpi  = zscore_global(d["cpi_yoy_change"])
    z_unem = zscore_global(d["unemployment_change_3m"])
    d["macro_stress_score"]    = z_rate + z_cpi + z_unem
    d["vol_macro_interaction"] = d["hvol_21d"] * d["macro_stress_score"]
    d["earnings_momentum"]     = d["last_eps_surprise_pct"] * d["earnings_streak"]
    d["rate_vol_regime"]       = d["fed_rate_change_3m"] * d["hvol_63d"]
    return d


def train_base_fold(X_tr, y_tr_enc):
    """Train both base models on a training fold.  Returns (xgb, lgbm)."""
    y_tr_orig = np.array([LABEL_DECODE[v] for v in y_tr_enc])
    sw = compute_sample_weight(class_weight="balanced", y=y_tr_orig)

    xgb = XGBClassifier(**XGB_PARAMS)
    xgb.fit(X_tr, y_tr_enc, sample_weight=sw)

    lgbm = LGBMClassifier(**LGBM_PARAMS)
    lgbm.fit(X_tr, y_tr_enc, sample_weight=sw)

    return xgb, lgbm


def make_meta_features(xgb_proba, lgbm_proba):
    """Concatenate base model probas -> 12-column meta-feature matrix."""
    return np.hstack([xgb_proba, lgbm_proba])


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Load & prepare data ────────────────────────────────────────────────────
    print("\nLoading data and building features ...")
    raw = pd.read_parquet(DATA_FILE)

    with open(FEAT_FILE) as f:
        feat_meta = json.load(f)
    tech_features = feat_meta["features"]

    raw["days_since_last_earnings"]      = raw["days_since_last_earnings"].fillna(S.CAP_EARNINGS)
    raw["days_since_last_product_event"] = raw["days_since_last_product_event"].fillna(S.CAP_PRODUCT)

    df = build_interaction_features(raw)
    all_features = tech_features + EVENT_FEATURES + INTERACTION_FEATURES

    df = df[all_features + [TARGET]].dropna()
    X  = df[all_features].values
    y  = df[TARGET].values

    print(f"  Samples  : {len(df):,}  ({df.index.min().date()} to {df.index.max().date()})")
    print(f"  Features : {len(all_features)}  (57: 36 tech + 16 event + 5 interaction)")
    print(f"  Target   : {TARGET}")

    counts = pd.Series(y).value_counts().sort_index()
    total  = len(y)
    for cls in CLASSES:
        n = counts.get(cls, 0)
        print(f"  {CLASS_NAMES[cls]}  {n:>5} ({n/total*100:.1f}%)")

    y_enc = np.array([LABEL_ENCODE[v] for v in y])

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Generate OOS probas from both base models across all 5 folds
    # ══════════════════════════════════════════════════════════════════════════
    section("PHASE 1 — BASE MODEL TRAINING  (5-fold walk-forward)")
    print("  Retraining XGBoost + LightGBM from scratch in each fold ...\n")

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    # Each fold produces: meta_X rows (12 cols), actual labels, date indices
    fold_meta_X   = []   # list of (n_test, 12) arrays — one per fold
    fold_actual   = []   # list of (n_test,) arrays
    fold_indices  = []   # list of test index arrays (positional, for df.index lookup)
    fold_xgb_acc  = []
    fold_lgbm_acc = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

        print(f"  Fold {fold}  |  "
              f"Train: {df.index[train_idx[0]].date()} - {df.index[train_idx[-1]].date()} ({len(train_idx):,})  |  "
              f"Test: {df.index[test_idx[0]].date()} - {df.index[test_idx[-1]].date()} ({len(test_idx):,})")

        xgb, lgbm = train_base_fold(X_tr, y_tr)

        xgb_proba  = xgb.predict_proba(X_te)    # (n_test, 3)
        lgbm_proba = lgbm.predict_proba(X_te)   # (n_test, 3)

        # Log per-model fold accuracy
        xgb_acc  = accuracy_score(y_te, xgb.predict(X_te))
        lgbm_acc = accuracy_score(y_te, lgbm.predict(X_te))
        fold_xgb_acc.append(xgb_acc)
        fold_lgbm_acc.append(lgbm_acc)
        print(f"           XGB acc={xgb_acc:.3f}   LGBM acc={lgbm_acc:.3f}")

        fold_meta_X.append(make_meta_features(xgb_proba, lgbm_proba))
        fold_actual.append(np.array([LABEL_DECODE[v] for v in y_te]))
        fold_indices.append(test_idx)

    print(f"\n  Base model fold accuracies:")
    print(f"  {'Fold':>4}  {'XGB':>7}  {'LGBM':>7}")
    for k in range(N_SPLITS):
        print(f"  {k+1:>4}  {fold_xgb_acc[k]:>7.3f}  {fold_lgbm_acc[k]:>7.3f}")
    print(f"  {'Mean':>4}  {np.mean(fold_xgb_acc):>7.3f}  {np.mean(fold_lgbm_acc):>7.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Walk-forward meta-learner training & evaluation
    # ══════════════════════════════════════════════════════════════════════════
    section("PHASE 2 — META-LEARNER  (walk-forward, folds 2-5)")
    print("  Protocol: meta trained on accumulated prior-fold OOS probas only")
    print("  (Fold 1 excluded — no prior meta training data available)\n")

    meta_col_names = [
        "xgb_bear", "xgb_side", "xgb_bull",
        "lgbm_bear", "lgbm_side", "lgbm_bull",
    ]

    meta_results    = []   # (fold, actual, predicted, proba, indices)
    fold_meta_res   = []   # summary row per fold

    for k in range(1, N_SPLITS):   # folds 2..5  (0-indexed: 1..4)
        # Accumulate all prior fold OOS meta-features as training data
        meta_train_X = np.vstack(fold_meta_X[:k])          # folds 0..k-1
        meta_train_y = np.concatenate(fold_actual[:k])     # original labels

        meta_test_X  = fold_meta_X[k]
        meta_test_y  = fold_actual[k]
        test_indices = fold_indices[k]

        meta_train_y_enc = np.array([LABEL_ENCODE[v] for v in meta_train_y])
        meta_test_y_enc  = np.array([LABEL_ENCODE[v] for v in meta_test_y])

        # Fit meta-learner
        meta_lr = LogisticRegressionCV(**META_PARAMS)
        meta_lr.fit(meta_train_X, meta_train_y_enc)

        # Predict
        meta_pred_enc   = meta_lr.predict(meta_test_X)
        meta_pred_proba = meta_lr.predict_proba(meta_test_X)
        meta_pred_dec   = np.array([LABEL_DECODE[v] for v in meta_pred_enc])

        fold_acc = accuracy_score(meta_test_y, meta_pred_dec)
        per_class = {
            cls: accuracy_score(
                meta_test_y[meta_test_y == cls],
                meta_pred_dec[meta_test_y == cls]
            ) if (meta_test_y == cls).sum() > 0 else float("nan")
            for cls in CLASSES
        }

        best_C = float(np.mean(list(meta_lr.C_)))
        print(f"  Fold {k+1}  |  meta_train={len(meta_train_X):,}  meta_test={len(meta_test_X):,}  "
              f"best_C={best_C:.4f}")
        print(f"         Accuracy: {fold_acc:.3f}   "
              f"Bear: {per_class[-1]:.3f}   "
              f"Side: {per_class[0]:.3f}   "
              f"Bull: {per_class[1]:.3f}")

        fold_meta_res.append({
            "fold": k + 1, "meta_train": len(meta_train_X),
            "meta_test": len(meta_test_X), "accuracy": fold_acc,
            "acc_bear": per_class[-1], "acc_side": per_class[0], "acc_bull": per_class[1],
        })

        meta_results.append({
            "actual"  : meta_test_y,
            "predicted": meta_pred_dec,
            "proba"   : meta_pred_proba,
            "indices" : test_indices,
        })

    # ══════════════════════════════════════════════════════════════════════════
    # Combined OOS evaluation (folds 2-5)
    # ══════════════════════════════════════════════════════════════════════════
    section("OUT-OF-SAMPLE RESULTS  (ensemble, folds 2–5 combined)")

    all_actual    = np.concatenate([r["actual"]    for r in meta_results])
    all_predicted = np.concatenate([r["predicted"] for r in meta_results])
    all_proba     = np.vstack([r["proba"]          for r in meta_results])
    all_indices   = np.concatenate([r["indices"]   for r in meta_results])

    oos_acc  = accuracy_score(all_actual, all_predicted)
    macro_f1 = f1_score(all_actual, all_predicted, labels=CLASSES, average="macro")
    majority_cls, naive_acc = naive_baseline_accuracy(all_actual)
    beats_naive = oos_acc > naive_acc

    per_class_recall = {
        cls: accuracy_score(all_actual[all_actual == cls], all_predicted[all_actual == cls])
        if (all_actual == cls).sum() > 0 else float("nan")
        for cls in CLASSES
    }

    print(f"\n  Overall OOS accuracy  : {oos_acc:.4f}  ({oos_acc*100:.2f}%)")
    print(f"  Naive baseline        : {naive_acc:.4f}  ({naive_acc*100:.2f}%)  "
          f"[always predict {CLASS_NAMES[majority_cls]}]")
    print(f"  Beats naive baseline  : "
          f"{'YES (+{:.2f}pp)'.format((oos_acc-naive_acc)*100) if beats_naive else 'NO ({:.2f}pp below)'.format((naive_acc-oos_acc)*100)}")
    print(f"  Macro F1              : {macro_f1:.4f}")

    print(f"\n  Per-class accuracy:")
    for cls in CLASSES:
        n = (all_actual == cls).sum()
        rec = per_class_recall[cls]
        print(f"    {CLASS_NAMES[cls]}  n={n:>5}  accuracy={rec:.4f}  ({rec*100:.1f}%)")

    print(f"\n  Classification report:\n")
    report = classification_report(
        all_actual, all_predicted,
        labels=CLASSES, target_names=CLASS_LABELS, digits=3,
    )
    for line in report.split("\n"):
        print(f"    {line}")

    print(f"  Confusion matrix (rows = actual, cols = predicted):\n")
    cm = confusion_matrix(all_actual, all_predicted, labels=CLASSES)
    print(f"  {'':12}{'Bear (-1)':>11}{'Side ( 0)':>11}{'Bull (+1)':>11}")
    print(f"  {'-'*45}")
    for i, cls in enumerate(CLASSES):
        print(f"  {CLASS_NAMES[cls]:<12}" + "".join(f"{cm[i,j]:>11}" for j in range(3)))

    print(f"\n  Per-fold meta summary (folds 2-5):")
    print(f"  {'Fold':>4}  {'MetaTrain':>9}  {'Test':>5}  {'Acc':>6}  "
          f"{'Bear':>6}  {'Side':>6}  {'Bull':>6}")
    print(f"  {'-'*55}")
    for r in fold_meta_res:
        print(f"  {r['fold']:>4}  {r['meta_train']:>9,}  {r['meta_test']:>5,}  "
              f"{r['accuracy']:>6.3f}  {r['acc_bear']:>6.3f}  "
              f"{r['acc_side']:>6.3f}  {r['acc_bull']:>6.3f}")
    meta_accs = [r["accuracy"] for r in fold_meta_res]
    print(f"  {'Mean':>4}  {'':>9}  {'':>5}  {np.mean(meta_accs):>6.3f}  "
          f"{np.mean([r['acc_bear']  for r in fold_meta_res]):>6.3f}  "
          f"{np.mean([r['acc_side']  for r in fold_meta_res]):>6.3f}  "
          f"{np.mean([r['acc_bull']  for r in fold_meta_res]):>6.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # Head-to-head comparison
    # ══════════════════════════════════════════════════════════════════════════
    section("FULL COMPARISON TABLE")

    name_map = {-1: "Bear", 0: "Sideways", 1: "Bull"}
    print(f"\n  {'Model':<35}  {'Acc':>7}  {'F1':>7}  {'Bear':>7}  {'Side':>7}  {'Bull':>7}")
    print(f"  {'-'*75}")

    rows = [
        ("XGBoost P3 champion (57 feat)",
         P3["acc"], P3["f1"],
         P3["recall"]["Bear"], P3["recall"]["Sideways"], P3["recall"]["Bull"]),
        ("LightGBM (57 feat)",
         P4L["acc"], P4L["f1"],
         P4L["recall"]["Bear"], P4L["recall"]["Sideways"], P4L["recall"]["Bull"]),
        ("Ensemble XGB+LGBM (folds 2-5)",
         oos_acc, macro_f1,
         per_class_recall[-1], per_class_recall[0], per_class_recall[1]),
    ]
    for label, acc, f1, bear, side, bull in rows:
        print(f"  {label:<35}  {acc*100:>6.2f}%  {f1:>7.3f}  "
              f"{bear*100:>6.1f}%  {side*100:>6.1f}%  {bull*100:>6.1f}%")

    print(f"\n  Naive baseline (always predict {CLASS_NAMES[majority_cls]}): {naive_acc*100:.2f}%")

    delta_f1  = macro_f1 - P3["f1"]
    delta_acc = (oos_acc - P3["acc"]) * 100
    beats_p3  = macro_f1 > P3["f1"]

    print(f"\n  Ensemble vs XGBoost P3 champion:")
    print(f"    Macro F1  : {P3['f1']:.3f} -> {macro_f1:.3f}  ({delta_f1:+.3f})")
    print(f"    Accuracy  : {P3['acc']*100:.2f}% -> {oos_acc*100:.2f}%  ({delta_acc:+.2f}pp)")
    for cls in CLASSES:
        name = name_map[cls]
        d = (per_class_recall[cls] - P3["recall"][name]) * 100
        print(f"    {name} recall : {P3['recall'][name]*100:.1f}% -> {per_class_recall[cls]*100:.1f}%  ({d:+.1f}pp)")

    # ══════════════════════════════════════════════════════════════════════════
    # Save predictions
    # ══════════════════════════════════════════════════════════════════════════
    section("SAVING OUTPUTS")

    pred_df = pd.DataFrame({
        "actual"       : all_actual,
        "predicted"    : all_predicted,
        "correct"      : (all_actual == all_predicted).astype(int),
        "prob_bear"    : all_proba[:, 0],
        "prob_sideways": all_proba[:, 1],
        "prob_bull"    : all_proba[:, 2],
        "confidence"   : all_proba.max(axis=1),
    }, index=df.index[all_indices])
    pred_df.index.name = "date"
    pred_df = pred_df.sort_index()

    PRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(PRED_FILE)
    print(f"\n  Predictions saved : {PRED_FILE.name}  ({len(pred_df):,} rows, folds 2-5)")

    # ══════════════════════════════════════════════════════════════════════════
    # Final model — retrain base models + meta on full data
    # ══════════════════════════════════════════════════════════════════════════
    section("TRAINING FINAL ENSEMBLE ON FULL DATASET")

    print("  Retraining XGBoost and LightGBM on full dataset ...")
    xgb_final, lgbm_final = train_base_fold(X, y_enc)

    # Meta-learner trained on ALL 5 folds of OOS meta-features (6,125 rows)
    all_meta_X = np.vstack(fold_meta_X)
    all_meta_y = np.concatenate(fold_actual)
    all_meta_y_enc = np.array([LABEL_ENCODE[v] for v in all_meta_y])

    print(f"  Training meta-learner on {len(all_meta_X):,} rows of OOS meta-features ...")
    meta_final = LogisticRegressionCV(**META_PARAMS)
    meta_final.fit(all_meta_X, all_meta_y_enc)
    best_C_final = float(np.mean(list(meta_final.C_)))
    print(f"  Meta-learner best C (mean across classes): {best_C_final:.4f}")

    # Persist as a bundle so inference only needs one file
    ensemble_bundle = {
        "xgb"          : xgb_final,
        "lgbm"         : lgbm_final,
        "meta"         : meta_final,
        "features"     : all_features,
        "target"       : TARGET,
        "meta_columns" : ["xgb_bear", "xgb_side", "xgb_bull",
                          "lgbm_bear", "lgbm_side", "lgbm_bull"],
        "label_encode" : LABEL_ENCODE,
        "label_decode" : LABEL_DECODE,
        "oos_acc"      : oos_acc,
        "macro_f1"     : macro_f1,
        "beats_xgb_p3" : beats_p3,
    }

    if beats_p3:
        ENSEMBLE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ENSEMBLE_FILE, "wb") as f:
            pickle.dump(ensemble_bundle, f)
        print(f"\n  Ensemble BEATS XGBoost P3 — saved as champion: {ENSEMBLE_FILE.name}")
    else:
        # Save anyway for Phase 4 Step 2 record, but do not promote to champion
        ENSEMBLE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ENSEMBLE_FILE, "wb") as f:
            pickle.dump(ensemble_bundle, f)
        print(f"\n  Ensemble below XGBoost P3 — saved for reference: {ENSEMBLE_FILE.name}")
        print(f"  XGBoost P3 champion remains: models/xgb_phase3_champion.pkl")

    # ══════════════════════════════════════════════════════════════════════════
    # Meta-learner coefficients
    # ══════════════════════════════════════════════════════════════════════════
    section("META-LEARNER COEFFICIENTS  (LogisticRegressionCV)")

    meta_col_names = [
        "xgb_prob_bear", "xgb_prob_side", "xgb_prob_bull",
        "lgbm_prob_bear", "lgbm_prob_side", "lgbm_prob_bull",
    ]

    print(f"\n  Best C (mean across classes): {best_C_final:.4f}")
    print(f"\n  Coefficients  (rows = predicted class, cols = meta-feature):")
    print(f"  {'Feature':<22}  {'->Bear':>8}  {'->Side':>8}  {'->Bull':>8}")
    print(f"  {'-'*52}")
    for j, fname in enumerate(meta_col_names):
        coefs = meta_final.coef_[:, j]   # one coef per output class
        print(f"  {fname:<22}  {coefs[0]:>+8.4f}  {coefs[1]:>+8.4f}  {coefs[2]:>+8.4f}")

    # Effective weighting: sum of absolute coefs for each base model
    xgb_weight  = np.abs(meta_final.coef_[:, :3]).mean()
    lgbm_weight = np.abs(meta_final.coef_[:, 3:]).mean()
    total_w     = xgb_weight + lgbm_weight
    print(f"\n  Effective model weights (mean |coef|):")
    print(f"    XGBoost  : {xgb_weight:.4f}  ({xgb_weight/total_w*100:.1f}%)")
    print(f"    LightGBM : {lgbm_weight:.4f}  ({lgbm_weight/total_w*100:.1f}%)")

    # ══════════════════════════════════════════════════════════════════════════
    # Final verdict and recommendations
    # ══════════════════════════════════════════════════════════════════════════
    section("PHASE 4 STEP 2 SUMMARY")

    if beats_p3:
        verdict = f"ENSEMBLE WINS  (F1 {P3['f1']:.3f} -> {macro_f1:.3f}, +{delta_f1:.3f})"
        champion_note = "New Phase 4 champion: ensemble_v1.pkl"
        next_steps = (
            "1. Update ManthIQ Model Lab to use ensemble predictions\n"
            "2. Begin Phase 5 (Prediction Engine) — multi-horizon outputs"
        )
    else:
        verdict = f"XGBoost P3 remains champion  (Ensemble F1={macro_f1:.3f} vs XGB {P3['f1']:.3f}, {delta_f1:.3f})"
        champion_note = "Champion unchanged: xgb_phase3_champion.pkl  (F1=0.375)"
        next_steps = (
            "Options to improve the ensemble:\n"
            "  A. Tune meta-learner — try higher C or add class weights\n"
            "  B. Tune LightGBM — num_leaves, min_child_samples, dart mode\n"
            "  C. Add a third base model — RandomForest or CatBoost\n"
            "  D. Add confidence gap as a meta-feature (|best_prob - 2nd_prob|)\n"
            "  E. Try soft-voting average instead of a learned meta-learner\n"
            "  Recommendation: Option D or E first (low cost, may close the gap)"
        )

    print(f"""
  Ensemble architecture:
    Base models : XGBoost (57 feat) + LightGBM (57 feat)
    Meta-learner: LogisticRegressionCV  (C={best_C_final:.4f}, multinomial)
    Meta-features: 12  (xgb_prob × 3 + lgbm_prob × 3 × 2)
    OOS window  : folds 2-5  ({len(all_actual):,} samples, {len(all_actual)/total*100:.0f}% of data)

  Results:
    OOS Accuracy : {oos_acc:.4f}  ({oos_acc*100:.2f}%)
    Macro F1     : {macro_f1:.4f}
    Naive base   : {naive_acc:.4f}  ({naive_acc*100:.2f}%)  ({'BEATS' if beats_naive else 'below'})

  Per-class recall vs XGBoost P3:
    Bear     : {per_class_recall[-1]*100:.1f}%  (XGB P3: {P3['recall']['Bear']*100:.1f}%  {(per_class_recall[-1]-P3['recall']['Bear'])*100:+.1f}pp)
    Sideways : {per_class_recall[0]*100:.1f}%  (XGB P3: {P3['recall']['Sideways']*100:.1f}%  {(per_class_recall[0]-P3['recall']['Sideways'])*100:+.1f}pp)
    Bull     : {per_class_recall[1]*100:.1f}%  (XGB P3: {P3['recall']['Bull']*100:.1f}%  {(per_class_recall[1]-P3['recall']['Bull'])*100:+.1f}pp)

  Verdict: {verdict}
  {champion_note}

  Next steps:
    {next_steps}

  Saved:
    {ENSEMBLE_FILE}
    {PRED_FILE}
""")
    print("Phase 4 Step 2 complete.\n")
