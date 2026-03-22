"""
Phase 2 - Experiment B: Class-weighted XGBoost on dir_1m
==========================================================
Same dir_1m target as baseline but with inverse-frequency sample weights
to counter the Bull-dominated class imbalance (Bull 51.9%, Bear 31.4%, Side 16.7%).

Sample weights: each sample gets weight = N / (n_classes * class_count)
so that all three classes contribute equally to the loss.

Target:  dir_1m  (+1 = Bull, 0 = Sideways, -1 = Bear)
Features: 36 clean features from models/feature_list.json

Outputs:
  models/xgb_dir_1m_weighted.pkl
  data/processed/aapl_predictions_1m_weighted.parquet
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from config import paths as P, settings as S

DATA_FILE  = P.DATA_LABELED
FEAT_FILE  = P.FEATURE_LIST
MODEL_FILE = P.MODEL_WEIGHTED_1M
PRED_FILE  = P.PRED_WEIGHTED_1M

TARGET      = "dir_1m"
N_SPLITS    = S.N_SPLITS
XGB_PARAMS  = S.XGB_PARAMS
LABEL_ENCODE = S.LABEL_ENCODE
LABEL_DECODE = S.LABEL_DECODE
CLASS_NAMES  = S.CLASS_NAMES
CLASSES      = S.CLASSES


def section(title):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


def naive_baseline_accuracy(y_true):
    counts = pd.Series(y_true).value_counts()
    majority_class = counts.idxmax()
    majority_frac  = counts.max() / len(y_true)
    return majority_class, majority_frac


if __name__ == "__main__":

    print("\nLoading data...")
    df = pd.read_parquet(DATA_FILE)
    with open(FEAT_FILE) as f:
        feat_meta = json.load(f)
    features = feat_meta["features"]

    cols_needed = features + [TARGET]
    df = df[cols_needed].dropna()

    X = df[features].values
    y = df[TARGET].values

    print(f"  Samples after dropna : {len(df):,}")
    print(f"  Features             : {len(features)}")
    print(f"  Target               : {TARGET}  (with class-balanced sample weights)")
    print(f"  Date range           : {df.index.min().date()} to {df.index.max().date()}")

    counts = pd.Series(y).value_counts().sort_index()
    total  = len(y)
    print(f"\n  Target distribution (before weighting):")
    for cls in CLASSES:
        n = counts.get(cls, 0)
        w = total / (3 * n) if n > 0 else 0
        print(f"    {CLASS_NAMES[cls]}  {n:>5} ({n/total*100:.1f}%)  sample_weight={w:.3f}")

    y_enc = np.array([LABEL_ENCODE[v] for v in y])

    section(f"WALK-FORWARD VALIDATION  ({N_SPLITS} expanding windows, class-balanced weights)")
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    fold_results  = []
    all_indices   = []
    all_predicted = []
    all_actual    = []
    all_proba     = []
    all_folds     = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        # Compute inverse-frequency sample weights from training fold only (no leakage)
        y_train_orig = np.array([LABEL_DECODE[v] for v in y_train])
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_orig)

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        acc     = accuracy_score(y_test, y_pred)

        y_test_dec = np.array([LABEL_DECODE[v] for v in y_test])
        y_pred_dec = np.array([LABEL_DECODE[v] for v in y_pred])

        per_class = {}
        for cls in CLASSES:
            mask = y_test_dec == cls
            if mask.sum() > 0:
                per_class[cls] = accuracy_score(y_test_dec[mask], y_pred_dec[mask])
            else:
                per_class[cls] = float("nan")

        fold_results.append({
            "fold"       : fold,
            "train_size" : len(train_idx),
            "test_size"  : len(test_idx),
            "accuracy"   : acc,
            **{f"acc_{CLASS_NAMES[c].split()[0].lower()}": per_class[c] for c in CLASSES},
        })

        print(f"\n  Fold {fold}  |  "
              f"Train: {df.index[train_idx[0]].date()} - {df.index[train_idx[-1]].date()} ({len(train_idx):,})  |  "
              f"Test: {df.index[test_idx[0]].date()} - {df.index[test_idx[-1]].date()} ({len(test_idx):,})")
        print(f"         Accuracy: {acc:.3f}   "
              f"Bear: {per_class[-1]:.3f}   "
              f"Side: {per_class[0]:.3f}   "
              f"Bull: {per_class[1]:.3f}")

        all_indices.extend(test_idx.tolist())
        all_predicted.extend(y_pred_dec.tolist())
        all_actual.extend(y_test_dec.tolist())
        all_proba.extend(y_proba.tolist())
        all_folds.extend([fold] * len(test_idx))

    section("OUT-OF-SAMPLE RESULTS  (all folds combined)")

    all_actual    = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    all_proba     = np.array(all_proba)

    oos_acc = accuracy_score(all_actual, all_predicted)
    majority_cls, naive_acc = naive_baseline_accuracy(all_actual)
    beats_naive = oos_acc > naive_acc

    print(f"\n  Overall OOS accuracy  : {oos_acc:.4f}  ({oos_acc*100:.2f}%)")
    print(f"  Naive baseline        : {naive_acc:.4f}  ({naive_acc*100:.2f}%)  [always predict {CLASS_NAMES[majority_cls]}]")
    print(f"  Beats naive baseline  : {'YES (+{:.2f}pp)'.format((oos_acc - naive_acc)*100) if beats_naive else 'NO ({:.2f}pp below)'.format((naive_acc - oos_acc)*100)}")

    print(f"\n  Per-class accuracy:")
    for cls in CLASSES:
        mask = all_actual == cls
        if mask.sum() > 0:
            cls_acc = accuracy_score(all_actual[mask], all_predicted[mask])
            n = mask.sum()
            print(f"    {CLASS_NAMES[cls]}  n={n:>5}  accuracy={cls_acc:.4f}  ({cls_acc*100:.1f}%)")

    print(f"\n  Classification report (precision / recall / f1):\n")
    report = classification_report(
        all_actual, all_predicted,
        labels=CLASSES,
        target_names=["Bear", "Sideways", "Bull"],
        digits=3,
    )
    for line in report.split("\n"):
        print(f"    {line}")

    print(f"  Confusion matrix (rows = actual, cols = predicted):\n")
    cm = confusion_matrix(all_actual, all_predicted, labels=CLASSES)
    print(f"  {'':12}{'Bear (-1)':>11}{'Side ( 0)':>11}{'Bull (+1)':>11}")
    print(f"  {'-'*45}")
    for i, cls in enumerate(CLASSES):
        print(f"  {CLASS_NAMES[cls]:<12}" + "".join(f"{cm[i,j]:>11}" for j in range(3)))

    print(f"\n  Per-fold summary:")
    print(f"  {'Fold':>4}  {'Train':>6}  {'Test':>5}  {'Acc':>6}  {'Bear':>6}  {'Side':>6}  {'Bull':>6}")
    print(f"  {'-'*50}")
    for r in fold_results:
        print(f"  {r['fold']:>4}  {r['train_size']:>6,}  {r['test_size']:>5,}  "
              f"{r['accuracy']:>6.3f}  "
              f"{r['acc_bear']:>6.3f}  "
              f"{r['acc_side']:>6.3f}  "
              f"{r['acc_bull']:>6.3f}")
    accs = [r['accuracy'] for r in fold_results]
    print(f"  {'Mean':>4}  {'':>6}  {'':>5}  {np.mean(accs):>6.3f}  "
          f"{np.mean([r['acc_bear'] for r in fold_results]):>6.3f}  "
          f"{np.mean([r['acc_side'] for r in fold_results]):>6.3f}  "
          f"{np.mean([r['acc_bull'] for r in fold_results]):>6.3f}")

    section("SAVING OUTPUTS")

    pred_df = pd.DataFrame({
        "actual"       : all_actual,
        "predicted"    : all_predicted,
        "correct"      : (all_actual == all_predicted).astype(int),
        "prob_bear"    : all_proba[:, 0],
        "prob_sideways": all_proba[:, 1],
        "prob_bull"    : all_proba[:, 2],
        "confidence"   : all_proba.max(axis=1),
        "fold"         : all_folds,
    }, index=df.index[all_indices])
    pred_df.index.name = "date"
    pred_df = pred_df.sort_index()

    PRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(PRED_FILE)
    print(f"\n  Predictions saved  : {PRED_FILE.name}  ({len(pred_df):,} rows)")

    print("\n  Training final model on full dataset...")
    y_all_orig = np.array([LABEL_DECODE[v] for v in y_enc])
    final_weights = compute_sample_weight(class_weight="balanced", y=y_all_orig)
    final_model = XGBClassifier(**XGB_PARAMS)
    final_model.fit(X, y_enc, sample_weight=final_weights)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(final_model, f)
    print(f"  Final model saved  : {MODEL_FILE.name}")

    section("FEATURE IMPORTANCES  (top 15 by gain)")
    importances = final_model.get_booster().get_score(importance_type="gain")
    imp_df = (
        pd.Series(importances, name="gain")
        .rename_axis("feature")
        .reset_index()
        .sort_values("gain", ascending=False)
    )
    fname_map = {f"f{i}": name for i, name in enumerate(features)}
    imp_df["feature"] = imp_df["feature"].map(fname_map)
    imp_df["gain"] = imp_df["gain"].round(1)

    print(f"\n  {'Rank':>4}  {'Feature':<25}  {'Gain':>10}")
    print(f"  {'-'*45}")
    for rank, (_, row) in enumerate(imp_df.head(15).iterrows(), 1):
        print(f"  {rank:>4}  {row['feature']:<25}  {row['gain']:>10,.1f}")

    print(f"\nExperiment B complete.")
    print(f"  Target       : {TARGET} (class-balanced sample weights)")
    print(f"  OOS Accuracy : {oos_acc:.4f} vs naive {naive_acc:.4f}  ({'BEATS' if beats_naive else 'BELOW'} baseline)")
    print(f"  Model        : {MODEL_FILE}")
    print(f"  Predictions  : {PRED_FILE}\n")
