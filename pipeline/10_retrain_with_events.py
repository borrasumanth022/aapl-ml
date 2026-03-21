"""
Phase 3 - Step 3: Retrain XGBoost with Event Features
======================================================
Loads aapl_with_events.parquet and trains on 36 technical + 16 event features
using the same walk-forward setup as the Phase 2 best model so results are
directly comparable.

Phase 2 best model baseline (xgb_dir_1w_weighted):
  OOS accuracy : 38.35%   Naive baseline : 37.50%
  Macro F1     : 0.367
  Per-class    : Bear=23.1%  Sideways=45.5%  Bull=41.8%

Target:   dir_1w  (+1=Bull, 0=Sideways, -1=Bear)
Features: 36 technical (feature_list.json) + 16 event features  = 52 total

NaN handling for partial-coverage event features:
  days_since_last_earnings      -> fill NaN with 90 (cap; pre-2005 rows)
  days_since_last_product_event -> fill NaN with 180 (cap; pre-2000 rows)
  has_earnings_data flag informs model which rows have no earnings history.

Outputs:
  models/xgb_dir_1w_events.pkl
  data/processed/aapl_predictions_events.parquet
  models/shap_summary_events.csv
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_FILE  = Path(__file__).parent.parent / "data" / "processed" / "aapl_with_events.parquet"
FEAT_FILE  = Path(__file__).parent.parent / "models" / "feature_list.json"
MODEL_FILE = Path(__file__).parent.parent / "models" / "xgb_dir_1w_events.pkl"
PRED_FILE  = Path(__file__).parent.parent / "data" / "processed" / "aapl_predictions_events.parquet"
SHAP_FILE  = Path(__file__).parent.parent / "models" / "shap_summary_events.csv"

TARGET   = "dir_1w"
N_SPLITS = 5

# Same params as Phase 2 best model — isolates the effect of the new features
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
CLASS_LABELS = ["Bear", "Sideways", "Bull"]
CLASSES      = [-1, 0, 1]

EVENT_FEATURES = [
    "days_to_next_earnings", "days_since_last_earnings", "has_earnings_data",
    "last_eps_surprise_pct", "earnings_streak",
    "fed_rate_level", "fed_rate_change_1m", "fed_rate_change_3m",
    "cpi_yoy_change", "unemployment_level", "unemployment_change_3m",
    "days_to_next_product_event", "days_since_last_product_event",
    "is_iphone_cycle", "rate_environment", "inflation_regime",
]

# Phase 2 best model results for direct comparison
P2_OOS_ACC  = 0.3835
P2_MACRO_F1 = 0.367
P2_RECALL   = {"Bear": 0.2310, "Sideways": 0.4549, "Bull": 0.4177}


def section(title):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


def naive_baseline_accuracy(y_true):
    counts = pd.Series(y_true).value_counts()
    return counts.idxmax(), counts.max() / len(y_true)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Load & prepare ─────────────────────────────────────────────────────────
    print("\nLoading data ...")
    df = pd.read_parquet(DATA_FILE)

    with open(FEAT_FILE) as f:
        feat_meta = json.load(f)
    tech_features = feat_meta["features"]
    all_features  = tech_features + EVENT_FEATURES

    # Fill sentinel values for partial-coverage event features
    df["days_since_last_earnings"]      = df["days_since_last_earnings"].fillna(90)
    df["days_since_last_product_event"] = df["days_since_last_product_event"].fillna(180)

    cols_needed = all_features + [TARGET]
    df = df[cols_needed].dropna()

    X = df[all_features].values
    y = df[TARGET].values

    print(f"  Samples after dropna  : {len(df):,}")
    print(f"  Tech features         : {len(tech_features)}")
    print(f"  Event features        : {len(EVENT_FEATURES)}")
    print(f"  Total features        : {len(all_features)}")
    print(f"  Target                : {TARGET}")
    print(f"  Date range            : {df.index.min().date()} to {df.index.max().date()}")

    counts = pd.Series(y).value_counts().sort_index()
    total  = len(y)
    print(f"\n  Target distribution:")
    for cls in CLASSES:
        n = counts.get(cls, 0)
        print(f"    {CLASS_NAMES[cls]}  {n:>5} ({n/total*100:.1f}%)")

    y_enc = np.array([LABEL_ENCODE[v] for v in y])

    # ── Walk-forward validation ────────────────────────────────────────────────
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
            per_class[cls] = (
                accuracy_score(y_test_dec[mask], y_pred_dec[mask])
                if mask.sum() > 0 else float("nan")
            )

        fold_results.append({
            "fold": fold, "train_size": len(train_idx), "test_size": len(test_idx),
            "accuracy": acc,
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

    # ── OOS evaluation ─────────────────────────────────────────────────────────
    section("OUT-OF-SAMPLE RESULTS  (all folds combined)")

    all_actual    = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    all_proba     = np.array(all_proba)

    oos_acc = accuracy_score(all_actual, all_predicted)
    macro_f1 = f1_score(all_actual, all_predicted, labels=CLASSES, average="macro")
    majority_cls, naive_acc = naive_baseline_accuracy(all_actual)
    beats_naive = oos_acc > naive_acc

    per_class_recall = {}
    for cls in CLASSES:
        mask = all_actual == cls
        if mask.sum() > 0:
            per_class_recall[cls] = accuracy_score(all_actual[mask], all_predicted[mask])

    print(f"\n  Overall OOS accuracy  : {oos_acc:.4f}  ({oos_acc*100:.2f}%)")
    print(f"  Naive baseline        : {naive_acc:.4f}  ({naive_acc*100:.2f}%)  "
          f"[always predict {CLASS_NAMES[majority_cls]}]")
    print(f"  Beats naive baseline  : "
          f"{'YES (+{:.2f}pp)'.format((oos_acc-naive_acc)*100) if beats_naive else 'NO ({:.2f}pp below)'.format((naive_acc-oos_acc)*100)}")
    print(f"  Macro F1              : {macro_f1:.4f}")

    print(f"\n  Per-class accuracy:")
    for cls in CLASSES:
        n = (all_actual == cls).sum()
        rec = per_class_recall.get(cls, float("nan"))
        print(f"    {CLASS_NAMES[cls]}  n={n:>5}  accuracy={rec:.4f}  ({rec*100:.1f}%)")

    print(f"\n  Classification report:\n")
    report = classification_report(
        all_actual, all_predicted, labels=CLASSES,
        target_names=CLASS_LABELS, digits=3,
    )
    for line in report.split("\n"):
        print(f"    {line}")

    print(f"  Confusion matrix (rows=actual, cols=predicted):\n")
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
              f"{r['accuracy']:>6.3f}  {r['acc_bear']:>6.3f}  "
              f"{r['acc_side']:>6.3f}  {r['acc_bull']:>6.3f}")
    accs = [r["accuracy"] for r in fold_results]
    print(f"  {'Mean':>4}  {'':>6}  {'':>5}  {np.mean(accs):>6.3f}  "
          f"{np.mean([r['acc_bear'] for r in fold_results]):>6.3f}  "
          f"{np.mean([r['acc_side'] for r in fold_results]):>6.3f}  "
          f"{np.mean([r['acc_bull'] for r in fold_results]):>6.3f}")

    # ── Direct comparison vs Phase 2 ───────────────────────────────────────────
    section("DIRECT COMPARISON vs PHASE 2 BEST MODEL")

    delta_acc    = (oos_acc   - P2_OOS_ACC)   * 100
    delta_f1     = (macro_f1  - P2_MACRO_F1)  * 100
    name_map     = {-1: "Bear", 0: "Sideways", 1: "Bull"}

    print(f"\n  {'Metric':<30}  {'Phase 2':>10}  {'Phase 3':>10}  {'Delta':>10}")
    print(f"  {'-'*65}")
    print(f"  {'OOS Accuracy':<30}  {P2_OOS_ACC*100:>9.2f}%  {oos_acc*100:>9.2f}%  "
          f"  {delta_acc:>+7.2f}pp")
    print(f"  {'Macro F1':<30}  {P2_MACRO_F1:>10.3f}  {macro_f1:>10.3f}  "
          f"  {delta_f1/100:>+10.3f}")
    for cls in CLASSES:
        name = name_map[cls]
        p2   = P2_RECALL[name]
        p3   = per_class_recall.get(cls, float("nan"))
        d    = (p3 - p2) * 100
        print(f"  {name+' Recall':<30}  {p2*100:>9.1f}%  {p3*100:>9.1f}%  "
              f"  {d:>+7.2f}pp")

    best_delta = max(
        (per_class_recall.get(cls, 0) - P2_RECALL[name_map[cls]]) * 100
        for cls in CLASSES
    )
    print(f"\n  Event features {'IMPROVED' if delta_f1 > 0 else 'DID NOT IMPROVE'} Macro F1 "
          f"({'+'if delta_f1>0 else ''}{delta_f1/100:.3f})")
    print(f"  Bear recall change: "
          f"{(per_class_recall.get(-1,0)-P2_RECALL['Bear'])*100:+.2f}pp  "
          f"({P2_RECALL['Bear']*100:.1f}% -> {per_class_recall.get(-1,0)*100:.1f}%)")

    # ── Save predictions ────────────────────────────────────────────────────────
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

    # ── Train final model on all data ───────────────────────────────────────────
    print("\n  Training final model on full dataset ...")
    y_orig        = np.array([LABEL_DECODE[v] for v in y_enc])
    final_weights = compute_sample_weight(class_weight="balanced", y=y_orig)
    final_model   = XGBClassifier(**XGB_PARAMS)
    final_model.fit(X, y_enc, sample_weight=final_weights)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(final_model, f)
    print(f"  Model saved        : {MODEL_FILE.name}")

    # ── XGBoost gain importance ─────────────────────────────────────────────────
    section("XGBOOST FEATURE IMPORTANCE  (gain, top 20)")

    importances = final_model.get_booster().get_score(importance_type="gain")
    fname_map   = {f"f{i}": name for i, name in enumerate(all_features)}
    imp_df = (
        pd.Series(importances, name="gain")
        .rename_axis("feature")
        .reset_index()
        .assign(feature=lambda d: d["feature"].map(fname_map))
        .sort_values("gain", ascending=False)
    )
    imp_df["is_event"] = imp_df["feature"].isin(EVENT_FEATURES)

    print(f"\n  {'Rank':>4}  {'Feature':<30}  {'Gain':>8}  {'Type'}")
    print(f"  {'-'*55}")
    for rank, (_, row) in enumerate(imp_df.head(20).iterrows(), 1):
        tag = "[EVENT]" if row["is_event"] else "       "
        print(f"  {rank:>4}  {row['feature']:<30}  {row['gain']:>8.1f}  {tag}")

    event_in_top10 = imp_df.head(10)[imp_df.head(10)["is_event"]]["feature"].tolist()
    print(f"\n  Event features in top 10 by gain: {len(event_in_top10)}")
    for f in event_in_top10:
        print(f"    - {f}")

    # ── SHAP analysis ────────────────────────────────────────────────────────────
    section("SHAP ANALYSIS  (TreeExplainer on final model)")

    print("\n  Computing SHAP values ...")
    explainer = shap.TreeExplainer(final_model)
    shap_raw  = explainer.shap_values(X)

    shap_arr = np.array(shap_raw)
    if shap_arr.ndim == 3 and shap_arr.shape[0] == len(X):
        shap_vals = [shap_arr[:, :, i] for i in range(shap_arr.shape[2])]
    elif shap_arr.ndim == 3 and shap_arr.shape[0] == 3:
        shap_vals = [shap_arr[i] for i in range(3)]
    else:
        shap_vals = shap_raw

    class_order = ["Bear", "Sideways", "Bull"]
    mean_abs = {
        cls: np.abs(shap_vals[i]).mean(axis=0)
        for i, cls in enumerate(class_order)
    }
    shap_df = pd.DataFrame(mean_abs, index=all_features)
    shap_df["mean_all_classes"] = shap_df.mean(axis=1)
    shap_df["is_event"] = shap_df.index.isin(EVENT_FEATURES)
    shap_df = shap_df.sort_values("mean_all_classes", ascending=False)

    print(f"\n  Top 20 features by mean |SHAP| (all classes):\n")
    print(f"  {'Rank':>4}  {'Feature':<30}  {'Bear':>7}  {'Side':>7}  {'Bull':>7}  {'Mean':>7}  Type")
    print(f"  {'-'*75}")
    for rank, (feat, row) in enumerate(shap_df.head(20).iterrows(), 1):
        tag = "[EVENT]" if row["is_event"] else "       "
        print(f"  {rank:>4}  {feat:<30}  "
              f"{row['Bear']:>7.4f}  {row['Sideways']:>7.4f}  "
              f"{row['Bull']:>7.4f}  {row['mean_all_classes']:>7.4f}  {tag}")

    # Event features that made the SHAP top 10
    shap_top10 = shap_df.head(10)
    event_shap_top10 = shap_top10[shap_top10["is_event"]].index.tolist()
    print(f"\n  Event features in SHAP top 10:")
    if event_shap_top10:
        for f in event_shap_top10:
            row = shap_df.loc[f]
            dominant = max(class_order, key=lambda c: row[c])
            print(f"    {f:<35} mean={row['mean_all_classes']:.4f}  "
                  f"(strongest for {dominant})")
    else:
        print("    None in top 10 — event features have marginal SHAP impact")

    # Full event-feature SHAP breakdown
    print(f"\n  All event features ranked by mean |SHAP|:")
    event_shap = shap_df[shap_df["is_event"]].sort_values("mean_all_classes", ascending=False)
    for rank, (feat, row) in enumerate(event_shap.iterrows(), 1):
        dominant = max(class_order, key=lambda c: row[c])
        print(f"    {rank:>2}. {feat:<35}  mean={row['mean_all_classes']:.4f}  "
              f"Bear={row['Bear']:.4f}  Side={row['Sideways']:.4f}  Bull={row['Bull']:.4f}  "
              f"({dominant})")

    SHAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(SHAP_FILE)
    print(f"\n  SHAP summary saved : {SHAP_FILE.name}")

    # ── Final summary ─────────────────────────────────────────────────────────
    section("PHASE 3 STEP 3 SUMMARY")
    print(f"""
  Model          : dir_1w + class-balanced weights + 16 event features
  Features       : 36 technical + 16 event = 52 total

  OOS Accuracy   : {oos_acc:.4f}  ({'+' if oos_acc>P2_OOS_ACC else ''}{(oos_acc-P2_OOS_ACC)*100:.2f}pp vs Phase 2)
  Macro F1       : {macro_f1:.4f}  ({'+' if macro_f1>P2_MACRO_F1 else ''}{macro_f1-P2_MACRO_F1:.3f} vs Phase 2)
  Naive baseline : {naive_acc:.4f}  ({'BEATS' if beats_naive else 'BELOW'})

  Per-class recall vs Phase 2:
    Bear     : {per_class_recall.get(-1,0)*100:.1f}%  ({(per_class_recall.get(-1,0)-P2_RECALL['Bear'])*100:+.1f}pp)
    Sideways : {per_class_recall.get(0,0)*100:.1f}%  ({(per_class_recall.get(0,0)-P2_RECALL['Sideways'])*100:+.1f}pp)
    Bull     : {per_class_recall.get(1,0)*100:.1f}%  ({(per_class_recall.get(1,0)-P2_RECALL['Bull'])*100:+.1f}pp)

  Event features in SHAP top 10 : {len(event_shap_top10)}
  Top event feature (SHAP)       : {event_shap.index[0] if len(event_shap) > 0 else 'n/a'}

  Saved:
    {MODEL_FILE}
    {PRED_FILE}
    {SHAP_FILE}
""")
    print("Phase 3 Step 3 complete.\n")
