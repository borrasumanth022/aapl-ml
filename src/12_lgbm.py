"""
Phase 4 - Step 1: LightGBM on 57 features
==========================================
Same 57-feature dataset and dir_1w target as the Phase 3 XGBoost champion,
but using LightGBM instead of XGBoost. All other variables held constant:
  - Same 5-fold walk-forward validation
  - Same class-balanced sample weights (computed per fold)
  - Same interaction features (built from aapl_with_events.parquet)

Purpose: determine whether LightGBM gives better macro F1 than XGBoost on
this dataset. If yes, proceed to the ensemble (Step 2). If roughly equal,
the ensemble still makes sense as the models will have different error
distributions.

Phase 3 champion baseline (XGBoost, dir_1w, 57 feat):
  OOS Accuracy : 38.30%   Naive baseline : 37.50%
  Macro F1     : 0.375
  Per-class    : Bear=30.6%  Sideways=39.9%  Bull=42.0%

LightGBM params: mirror XGBoost conservatively
  num_leaves=31, max_depth=4, learning_rate=0.05, n_estimators=300
  subsample=0.8, colsample_bytree=0.8, min_child_samples=20

Outputs:
  models/lgbm_dir_1w.pkl
  data/processed/aapl_predictions_lgbm.parquet
  models/shap_summary_lgbm.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import pickle

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from config import paths as P, settings as S

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_FILE  = P.DATA_WITH_EVENTS
FEAT_FILE  = P.FEATURE_LIST
MODEL_FILE = P.MODEL_LGBM
PRED_FILE  = P.PRED_LGBM
SHAP_FILE  = P.ROOT / "models" / "shap_summary_lgbm.csv"

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET      = "dir_1w"
N_SPLITS    = S.N_SPLITS
LABEL_ENCODE = S.LABEL_ENCODE
LABEL_DECODE = S.LABEL_DECODE
CLASS_NAMES  = S.CLASS_NAMES
CLASS_LABELS = S.CLASS_LABELS
CLASSES      = S.CLASSES
EVENT_FEATURES       = S.EVENT_FEATURES
INTERACTION_FEATURES = S.INTERACTION_FEATURES

# LightGBM params — deliberately mirrors XGBoost config to isolate algo effect.
# num_leaves=31 is LightGBM's default (analogous to max_depth=4 trees).
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

# Phase 3 champion for direct comparison
P3 = S.PHASE3_CHAMPION


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


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Load & prepare ─────────────────────────────────────────────────────────
    print("\nLoading data and building interaction features ...")
    raw = pd.read_parquet(DATA_FILE)

    with open(FEAT_FILE) as f:
        feat_meta = json.load(f)
    tech_features = feat_meta["features"]

    # Sentinel fills for partial-coverage event features
    raw["days_since_last_earnings"]      = raw["days_since_last_earnings"].fillna(S.CAP_EARNINGS)
    raw["days_since_last_product_event"] = raw["days_since_last_product_event"].fillna(S.CAP_PRODUCT)

    df = build_interaction_features(raw)
    all_features = tech_features + EVENT_FEATURES + INTERACTION_FEATURES

    missing = [f for f in all_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df = df[all_features + [TARGET]].dropna()

    X = df[all_features].values
    y = df[TARGET].values

    print(f"  Samples after dropna  : {len(df):,}")
    print(f"  Tech features         : {len(tech_features)}")
    print(f"  Event features        : {len(EVENT_FEATURES)}")
    print(f"  Interaction features  : {len(INTERACTION_FEATURES)}")
    print(f"  Total features        : {len(all_features)}")
    print(f"  Target                : {TARGET}")
    print(f"  Date range            : {df.index.min().date()} to {df.index.max().date()}")

    counts = pd.Series(y).value_counts().sort_index()
    total  = len(y)
    print(f"\n  Target distribution:")
    for cls in CLASSES:
        n = counts.get(cls, 0)
        print(f"    {CLASS_NAMES[cls]}  {n:>5} ({n/total*100:.1f}%)")

    # LightGBM uses original integer labels directly (no encode/decode needed)
    # but we use encoded (0,1,2) to be consistent with XGBoost setup
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

        # Class-balanced weights from training fold only (no leakage)
        y_train_orig   = np.array([LABEL_DECODE[v] for v in y_train])
        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train_orig)

        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)   # shape (n, 3): [bear, side, bull]
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
              f"Test:  {df.index[test_idx[0]].date()} - {df.index[test_idx[-1]].date()} ({len(test_idx):,})")
        print(f"         Accuracy: {acc:.3f}   "
              f"Bear: {per_class[-1]:.3f}   "
              f"Side: {per_class[0]:.3f}   "
              f"Bull: {per_class[1]:.3f}")

        all_indices.extend(test_idx.tolist())
        all_predicted.extend(y_pred_dec.tolist())
        all_actual.extend(y_test_dec.tolist())
        all_proba.extend(y_proba.tolist())
        all_folds.extend([fold] * len(test_idx))

    # ── Combined OOS evaluation ────────────────────────────────────────────────
    section("OUT-OF-SAMPLE RESULTS  (all folds combined)")

    all_actual    = np.array(all_actual)
    all_predicted = np.array(all_predicted)
    all_proba     = np.array(all_proba)

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

    print(f"\n  Classification report (precision / recall / f1):\n")
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

    # ── Direct comparison vs Phase 3 champion ─────────────────────────────────
    section("DIRECT COMPARISON vs PHASE 3 CHAMPION  (XGBoost, 57 feat)")

    delta_acc = (oos_acc  - P3["acc"]) * 100
    delta_f1  = (macro_f1 - P3["f1"])
    name_map  = {-1: "Bear", 0: "Sideways", 1: "Bull"}

    print(f"\n  {'Metric':<28}  {'XGB P3':>10}  {'LGBM':>10}  {'Delta':>10}")
    print(f"  {'-'*65}")
    print(f"  {'OOS Accuracy':<28}  {P3['acc']*100:>9.2f}%  {oos_acc*100:>9.2f}%  "
          f"{delta_acc:>+9.2f}pp")
    print(f"  {'Macro F1':<28}  {P3['f1']:>10.3f}  {macro_f1:>10.3f}  "
          f"{delta_f1:>+10.3f}")
    print(f"  {'Naive baseline':<28}  {'37.50%':>10}  {'':>10}")
    for cls in CLASSES:
        name = name_map[cls]
        p3r  = P3["recall"][name]
        lgr  = per_class_recall[cls]
        d    = (lgr - p3r) * 100
        print(f"  {name+' Recall':<28}  {p3r*100:>9.1f}%  {lgr*100:>9.1f}%  "
              f"{d:>+9.2f}pp")

    if macro_f1 > P3["f1"]:
        verdict = f"LGBM BEATS XGBoost P3 champion  (+{delta_f1:.3f} F1)"
    elif macro_f1 == P3["f1"]:
        verdict = "LGBM ties XGBoost P3 champion"
    else:
        verdict = f"LGBM below XGBoost P3 champion  ({delta_f1:.3f} F1)"
    print(f"\n  Verdict: {verdict}")
    print(f"  Both models will enter the Phase 4 ensemble regardless of solo F1 result,")
    print(f"  as complementary probability outputs are the ensemble's primary value.")

    # ── Save predictions ───────────────────────────────────────────────────────
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

    # ── Train final model on full dataset ──────────────────────────────────────
    print("\n  Training final model on full dataset ...")
    y_all_orig    = np.array([LABEL_DECODE[v] for v in y_enc])
    final_weights = compute_sample_weight(class_weight="balanced", y=y_all_orig)
    final_model   = LGBMClassifier(**LGBM_PARAMS)
    final_model.fit(X, y_enc, sample_weight=final_weights)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(final_model, f)
    print(f"  Model saved        : {MODEL_FILE.name}")

    # ── LightGBM built-in feature importance (gain) ────────────────────────────
    section("LIGHTGBM FEATURE IMPORTANCE  (gain, top 20)")

    imp_series = pd.Series(
        final_model.booster_.feature_importance(importance_type="gain"),
        index=all_features,
        name="gain",
    ).sort_values(ascending=False)

    imp_df = imp_series.reset_index()
    imp_df.columns = ["feature", "gain"]
    imp_df["type"] = imp_df["feature"].map(
        lambda f: "[INTER]" if f in INTERACTION_FEATURES
        else "[EVENT]" if f in EVENT_FEATURES
        else "       "
    )

    print(f"\n  {'Rank':>4}  {'Feature':<32}  {'Gain':>10}  Type")
    print(f"  {'-'*58}")
    for rank, (_, row) in enumerate(imp_df.head(20).iterrows(), 1):
        print(f"  {rank:>4}  {row['feature']:<32}  {row['gain']:>10,.1f}  {row['type']}")

    # ── SHAP analysis ──────────────────────────────────────────────────────────
    section("SHAP ANALYSIS  (TreeExplainer on final model)")

    print("\n  Computing SHAP values ...")
    explainer = shap.TreeExplainer(final_model)
    shap_raw  = explainer.shap_values(X)

    # LightGBM + SHAP: shap_values returns list of (n_samples, n_features) per class
    shap_arr = np.array(shap_raw)
    if shap_arr.ndim == 3 and shap_arr.shape[0] == len(X):
        # shape: (n_samples, n_features, n_classes)
        sv = [shap_arr[:, :, i] for i in range(shap_arr.shape[2])]
    elif shap_arr.ndim == 3 and shap_arr.shape[0] == 3:
        # shape: (n_classes, n_samples, n_features)
        sv = [shap_arr[i] for i in range(3)]
    else:
        sv = shap_raw  # already a list

    class_order = ["Bear", "Sideways", "Bull"]
    mean_abs = {cls: np.abs(sv[i]).mean(axis=0) for i, cls in enumerate(class_order)}
    shap_df = pd.DataFrame(mean_abs, index=all_features)
    shap_df["mean_all_classes"] = shap_df.mean(axis=1)
    shap_df["type"] = shap_df.index.map(
        lambda f: "[INTER]" if f in INTERACTION_FEATURES
        else "[EVENT]" if f in EVENT_FEATURES
        else "       "
    )
    shap_df = shap_df.sort_values("mean_all_classes", ascending=False)

    print(f"\n  Top 20 features by mean |SHAP| (all classes):\n")
    print(f"  {'Rank':>4}  {'Feature':<32}  {'Bear':>7}  {'Side':>7}  {'Bull':>7}  {'Mean':>7}  Type")
    print(f"  {'-'*80}")
    for rank, (feat, row) in enumerate(shap_df.head(20).iterrows(), 1):
        print(f"  {rank:>4}  {feat:<32}  "
              f"{row['Bear']:>7.4f}  {row['Sideways']:>7.4f}  "
              f"{row['Bull']:>7.4f}  {row['mean_all_classes']:>7.4f}  {row['type']}")

    inter_shap = shap_df[shap_df["type"] == "[INTER]"]
    event_shap = shap_df[shap_df["type"] == "[EVENT]"]
    print(f"\n  Interaction features (SHAP rank / mean):")
    for feat, row in inter_shap.iterrows():
        rank = shap_df.index.get_loc(feat) + 1
        dominant = max(class_order, key=lambda c: row[c])
        print(f"    {feat:<35}  rank #{rank:<3}  mean={row['mean_all_classes']:.4f}  ({dominant})")

    print(f"\n  Top event features (SHAP, top 8):")
    for rank_e, (feat, row) in enumerate(event_shap.head(8).iterrows(), 1):
        rank_all = shap_df.index.get_loc(feat) + 1
        dominant = max(class_order, key=lambda c: row[c])
        print(f"    {rank_e}. {feat:<35}  rank #{rank_all:<3}  mean={row['mean_all_classes']:.4f}  ({dominant})")

    SHAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(SHAP_FILE)
    print(f"\n  SHAP summary saved : {SHAP_FILE.name}")

    # ── Final summary ──────────────────────────────────────────────────────────
    section("PHASE 4 STEP 1 SUMMARY")

    beat_p3 = macro_f1 > P3["f1"]
    print(f"""
  Model          : LightGBM dir_1w, class-balanced weights
  Features       : 57  (36 tech + 16 event + 5 interaction)
  Params         : num_leaves=31, max_depth=4, lr=0.05, n_est=300

  OOS Accuracy   : {oos_acc:.4f}  ({oos_acc*100:.2f}%)
  Macro F1       : {macro_f1:.4f}  ({'BEATS' if beat_p3 else 'below'} XGBoost P3 {P3['f1']:.3f}  delta={delta_f1:+.3f})
  Naive baseline : {naive_acc:.4f}  ({naive_acc*100:.2f}%)  ({'BEATS' if beats_naive else 'below'})

  Per-class recall vs XGBoost P3:
    Bear     : {per_class_recall[-1]*100:.1f}%  (XGB: {P3['recall']['Bear']*100:.1f}%  {(per_class_recall[-1]-P3['recall']['Bear'])*100:+.1f}pp)
    Sideways : {per_class_recall[0]*100:.1f}%  (XGB: {P3['recall']['Sideways']*100:.1f}%  {(per_class_recall[0]-P3['recall']['Sideways'])*100:+.1f}pp)
    Bull     : {per_class_recall[1]*100:.1f}%  (XGB: {P3['recall']['Bull']*100:.1f}%  {(per_class_recall[1]-P3['recall']['Bull'])*100:+.1f}pp)

  Top SHAP feature (overall) : {shap_df.index[0]}
  Top SHAP — Bear            : {shap_df.sort_values('Bear', ascending=False).index[0]}
  Top SHAP — Sideways        : {shap_df.sort_values('Sideways', ascending=False).index[0]}
  Top SHAP — Bull            : {shap_df.sort_values('Bull', ascending=False).index[0]}

  Saved:
    {MODEL_FILE}
    {PRED_FILE}
    {SHAP_FILE}

  Next step: src/13_ensemble.py — XGBoost + LightGBM meta-learner (Phase 4 Step 2)
""")
    print("Phase 4 Step 1 complete.\n")
