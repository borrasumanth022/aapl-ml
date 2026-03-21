"""
Phase 2 - Step 3: Best Model  (dir_1w + class-balanced weights)
================================================================
Combines the two experiment insights:
  - dir_1w target (beats naive, balanced classes, better Sideways recall)
  - Class-balanced sample weights (inverse-frequency, computed per fold)

Then runs SHAP TreeExplainer on the final model to identify which features
are actually driving predictions vs. just correlating with price trend.

Outputs:
  models/xgb_dir_1w_weighted.pkl            -- Phase 2 best model
  data/processed/aapl_predictions_best.parquet  -- OOS predictions + proba
  models/shap_summary.csv                   -- mean |SHAP| per feature (all classes)
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_FILE  = Path(__file__).parent.parent / "data" / "processed" / "aapl_labeled.parquet"
FEAT_FILE  = Path(__file__).parent.parent / "models" / "feature_list.json"
MODEL_FILE = Path(__file__).parent.parent / "models" / "xgb_dir_1w_weighted.pkl"
PRED_FILE  = Path(__file__).parent.parent / "data" / "processed" / "aapl_predictions_best.parquet"
SHAP_FILE  = Path(__file__).parent.parent / "models" / "shap_summary.csv"

TARGET   = "dir_1w"
N_SPLITS = 5

XGB_PARAMS = {
    "n_estimators"    : 300,
    "max_depth"       : 4,
    "learning_rate"   : 0.05,
    "subsample"       : 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "eval_metric"     : "mlogloss",
    "random_state"    : 42,
    "n_jobs"          : -1,
    "verbosity"       : 0,
}

LABEL_ENCODE = {-1: 0, 0: 1, 1: 2}
LABEL_DECODE = { 0:-1, 1: 0, 2: 1}
CLASS_NAMES  = {-1: "Bear (-1)", 0: "Side ( 0)", 1: "Bull (+1)"}
CLASS_LABELS = ["Bear", "Sideways", "Bull"]   # for SHAP output columns
CLASSES      = [-1, 0, 1]


def section(title):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


def naive_baseline_accuracy(y_true):
    counts = pd.Series(y_true).value_counts()
    return counts.idxmax(), counts.max() / len(y_true)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Load ───────────────────────────────────────────────────────────────────
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
    print(f"  Target               : {TARGET}  (class-balanced sample weights)")
    print(f"  Date range           : {df.index.min().date()} to {df.index.max().date()}")

    counts = pd.Series(y).value_counts().sort_index()
    total  = len(y)
    print(f"\n  Target distribution + effective weights:")
    for cls in CLASSES:
        n = counts.get(cls, 0)
        w = total / (3 * n) if n > 0 else 0
        print(f"    {CLASS_NAMES[cls]}  {n:>5} ({n/total*100:.1f}%)  weight={w:.3f}")

    y_enc = np.array([LABEL_ENCODE[v] for v in y])

    # ── Walk-forward validation ─────────────────────────────────────────────────
    section(f"WALK-FORWARD VALIDATION  ({N_SPLITS} expanding windows, class-balanced)")
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

        # Weights computed from training fold only (no leakage into test)
        y_train_orig   = np.array([LABEL_DECODE[v] for v in y_train])
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
            per_class[cls] = accuracy_score(y_test_dec[mask], y_pred_dec[mask]) if mask.sum() > 0 else float("nan")

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

    # ── Combined OOS evaluation ─────────────────────────────────────────────────
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
        target_names=CLASS_LABELS,
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

    # ── Save predictions ─────────────────────────────────────────────────────────
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

    # ── Train final model on all data ─────────────────────────────────────────────
    print("\n  Training final model on full dataset...")
    y_orig = np.array([LABEL_DECODE[v] for v in y_enc])
    final_weights = compute_sample_weight(class_weight="balanced", y=y_orig)
    final_model = XGBClassifier(**XGB_PARAMS)
    final_model.fit(X, y_enc, sample_weight=final_weights)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(final_model, f)
    print(f"  Final model saved  : {MODEL_FILE.name}")

    # ── XGBoost built-in feature importance (gain) ────────────────────────────────
    section("XGBOOST FEATURE IMPORTANCE  (gain, top 15)")

    importances = final_model.get_booster().get_score(importance_type="gain")
    imp_df = (
        pd.Series(importances, name="gain")
        .rename_axis("feature")
        .reset_index()
        .sort_values("gain", ascending=False)
    )
    fname_map = {f"f{i}": name for i, name in enumerate(features)}
    imp_df["feature"] = imp_df["feature"].map(fname_map)
    imp_df["gain"] = imp_df["gain"].round(2)

    print(f"\n  {'Rank':>4}  {'Feature':<25}  {'Gain':>10}")
    print(f"  {'-'*45}")
    for rank, (_, row) in enumerate(imp_df.head(15).iterrows(), 1):
        print(f"  {rank:>4}  {row['feature']:<25}  {row['gain']:>10,.2f}")

    # ── SHAP analysis ─────────────────────────────────────────────────────────────
    section("SHAP ANALYSIS  (TreeExplainer on final model)")

    print("\n  Computing SHAP values (this may take ~30s)...")
    explainer   = shap.TreeExplainer(final_model)
    shap_raw    = explainer.shap_values(X)
    # shap_raw shape depends on SHAP version:
    #   older: list of 3 arrays (n_samples, n_features)
    #   newer (0.40+): 3D array (n_samples, n_features, n_classes)
    shap_arr = np.array(shap_raw)  # force to ndarray for uniform handling
    if shap_arr.ndim == 3 and shap_arr.shape[0] == len(X):
        # shape: (n_samples, n_features, n_classes)
        shap_values = [shap_arr[:, :, i] for i in range(shap_arr.shape[2])]
    elif shap_arr.ndim == 3 and shap_arr.shape[0] == 3:
        # shape: (n_classes, n_samples, n_features)
        shap_values = [shap_arr[i] for i in range(3)]
    else:
        shap_values = shap_raw  # already a list

    class_order = ["Bear", "Sideways", "Bull"]

    # Mean absolute SHAP per feature per class
    mean_abs = {}
    for i, cls_label in enumerate(class_order):
        mean_abs[cls_label] = np.abs(shap_values[i]).mean(axis=0)

    shap_df = pd.DataFrame(mean_abs, index=features)
    shap_df["mean_all_classes"] = shap_df.mean(axis=1)
    shap_df = shap_df.sort_values("mean_all_classes", ascending=False)

    print(f"\n  Top 20 features by mean |SHAP| (averaged across all 3 classes):\n")
    print(f"  {'Rank':>4}  {'Feature':<25}  {'Bear':>8}  {'Sideways':>10}  {'Bull':>8}  {'Mean':>8}")
    print(f"  {'-'*70}")
    for rank, (feat, row) in enumerate(shap_df.head(20).iterrows(), 1):
        print(f"  {rank:>4}  {feat:<25}  "
              f"{row['Bear']:>8.4f}  "
              f"{row['Sideways']:>10.4f}  "
              f"{row['Bull']:>8.4f}  "
              f"{row['mean_all_classes']:>8.4f}")

    # Per-class top 10
    for cls_label in class_order:
        top10 = shap_df.sort_values(cls_label, ascending=False).head(10)
        print(f"\n  Top 10 features for {cls_label} predictions (mean |SHAP|):")
        for rank, (feat, row) in enumerate(top10.iterrows(), 1):
            print(f"    {rank:>2}. {feat:<25}  {row[cls_label]:.4f}")

    # Notable: features that matter a lot for one class but not others (specialised)
    print(f"\n  Class-specialised features  (top 5 that differ most across classes):")
    shap_df["spread"] = shap_df[class_order].max(axis=1) - shap_df[class_order].min(axis=1)
    specialized = shap_df.nlargest(5, "spread")
    for feat, row in specialized.iterrows():
        dominant = max(class_order, key=lambda c: row[c])
        print(f"    {feat:<25}  Bear={row['Bear']:.4f}  Side={row['Sideways']:.4f}  Bull={row['Bull']:.4f}  "
              f"(strongest for {dominant})")

    # Save SHAP summary
    SHAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(SHAP_FILE)
    print(f"\n  SHAP summary saved : {SHAP_FILE.name}")

    # ── Final summary ─────────────────────────────────────────────────────────────
    section("PHASE 2 BEST MODEL SUMMARY")
    print(f"""
  Model        : dir_1w + class-balanced sample weights
  OOS Accuracy : {oos_acc:.4f}  ({oos_acc*100:.2f}%)
  Naive base   : {naive_acc:.4f}  ({naive_acc*100:.2f}%)
  vs Naive     : {'BEATS by +{:.2f}pp'.format((oos_acc - naive_acc)*100) if beats_naive else 'BELOW by {:.2f}pp'.format((naive_acc - oos_acc)*100)}
  Macro F1     : see classification report above

  Key SHAP findings:
    Top overall driver : {shap_df.index[0]}
    Top Bear driver    : {shap_df.sort_values('Bear', ascending=False).index[0]}
    Top Side driver    : {shap_df.sort_values('Sideways', ascending=False).index[0]}
    Top Bull driver    : {shap_df.sort_values('Bull', ascending=False).index[0]}

  Saved artefacts:
    {MODEL_FILE}
    {PRED_FILE}
    {SHAP_FILE}
""")
    print("Phase 2 Step 3 complete. Ready for Phase 3 (event features).\n")
