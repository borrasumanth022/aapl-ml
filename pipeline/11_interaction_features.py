"""
Phase 3 - Step 4: Interaction Features + dir_1w vs dir_1m Comparison
=====================================================================
Builds 5 interaction features from the event + technical feature set,
then trains two models for direct comparison:

  Model A: dir_1w  (weekly direction, same as Phase 2 best)
  Model B: dir_1m  (monthly direction, macro context may help more)

Both use: 36 tech + 16 event + 5 interaction = 57 features
Same class-balanced weights + 5-fold walk-forward throughout.

Phase 2 baselines (for comparison):
  dir_1w : OOS 38.35%  Macro F1 0.367  (xgb_dir_1w_weighted)
  dir_1m : OOS 40.96%  Macro F1 0.343  (xgb_dir_1m_weighted)
  Phase 3 dir_1w (raw events, no interaction): 36.28%  F1 0.354

Interaction features:
  earnings_proximity_surprise  days_to_next_earnings * |last_eps_surprise_pct|
                               High = big known surprise + earnings close/far
  macro_stress_score           z-score sum of fed_rate_change_3m + cpi_yoy_change
                               + unemployment_change_3m -> single macro stress index
  vol_macro_interaction        hvol_21d * macro_stress_score
                               Volatility spike coinciding with macro stress
  earnings_momentum            last_eps_surprise_pct * earnings_streak
                               Magnitude of most recent surprise * streak consistency
  rate_vol_regime              fed_rate_change_3m * hvol_63d
                               Rate change in context of longer-term vol regime

Outputs:
  models/xgb_best_interactions.pkl       -- whichever of A/B wins
  data/processed/aapl_predictions_interactions.parquet
  models/shap_summary_interactions.csv
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_FILE  = Path(__file__).parent.parent / "data" / "processed" / "aapl_with_events.parquet"
FEAT_FILE  = Path(__file__).parent.parent / "models" / "feature_list.json"
MODEL_FILE = Path(__file__).parent.parent / "models" / "xgb_best_interactions.pkl"
PRED_FILE  = Path(__file__).parent.parent / "data" / "processed" / "aapl_predictions_interactions.parquet"
SHAP_FILE  = Path(__file__).parent.parent / "models" / "shap_summary_interactions.csv"

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
INTERACTION_FEATURES = [
    "earnings_proximity_surprise",
    "macro_stress_score",
    "vol_macro_interaction",
    "earnings_momentum",
    "rate_vol_regime",
]

# Phase 2 baselines
P2 = {
    "dir_1w": {"acc": 0.3835, "f1": 0.367,
               "recall": {"Bear": 0.2310, "Sideways": 0.4549, "Bull": 0.4177}},
    "dir_1m": {"acc": 0.4096, "f1": 0.3433,
               "recall": {"Bear": 0.2828, "Sideways": 0.1947, "Bull": 0.5525}},
}


def section(title):
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print(f"{'=' * 62}")


def zscore_global(s: pd.Series) -> pd.Series:
    """Global z-score — scales without using future label information."""
    return (s - s.mean()) / (s.std() + 1e-9)


def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 5 interaction features and append to df (in-place copy)."""
    d = df.copy()

    # 1. earnings_proximity_surprise
    #    Inverse proximity × |surprise|: higher = big surprise is imminent
    #    (90 - days_to_next) so score increases as earnings approach
    days_remaining = (90 - d["days_to_next_earnings"]).clip(lower=0)
    d["earnings_proximity_surprise"] = (
        days_remaining * d["last_eps_surprise_pct"].abs()
    )

    # 2. macro_stress_score  (z-score sum — equal-weighted composite)
    z_rate = zscore_global(d["fed_rate_change_3m"])
    z_cpi  = zscore_global(d["cpi_yoy_change"])
    z_unem = zscore_global(d["unemployment_change_3m"])
    d["macro_stress_score"] = z_rate + z_cpi + z_unem

    # 3. vol_macro_interaction
    d["vol_macro_interaction"] = d["hvol_21d"] * d["macro_stress_score"]

    # 4. earnings_momentum  (magnitude × streak consistency)
    d["earnings_momentum"] = d["last_eps_surprise_pct"] * d["earnings_streak"]

    # 5. rate_vol_regime  (rate change in context of longer-term vol)
    d["rate_vol_regime"] = d["fed_rate_change_3m"] * d["hvol_63d"]

    return d


def naive_baseline_accuracy(y_true):
    counts = pd.Series(y_true).value_counts()
    return counts.idxmax(), counts.max() / len(y_true)


def run_walkforward(X, y, df_index, label):
    """Run 5-fold walk-forward, return (all_actual, all_predicted, all_proba,
    all_indices, all_folds, fold_results)."""
    y_enc = np.array([LABEL_ENCODE[v] for v in y])
    tscv  = TimeSeriesSplit(n_splits=N_SPLITS)

    fold_results, all_idx, all_pred, all_act, all_proba, all_folds = [], [], [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]

        y_tr_orig = np.array([LABEL_DECODE[v] for v in y_tr])
        sw = compute_sample_weight(class_weight="balanced", y=y_tr_orig)

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_tr, y_tr, sample_weight=sw)

        y_pred  = model.predict(X_te)
        y_proba = model.predict_proba(X_te)
        acc     = accuracy_score(y_te, y_pred)

        y_te_d  = np.array([LABEL_DECODE[v] for v in y_te])
        y_pred_d = np.array([LABEL_DECODE[v] for v in y_pred])

        per_class = {
            cls: accuracy_score(y_te_d[y_te_d == cls], y_pred_d[y_te_d == cls])
            if (y_te_d == cls).sum() > 0 else float("nan")
            for cls in CLASSES
        }

        fold_results.append({
            "fold": fold, "train_size": len(train_idx), "test_size": len(test_idx),
            "accuracy": acc,
            **{f"acc_{CLASS_NAMES[c].split()[0].lower()}": per_class[c] for c in CLASSES},
        })
        print(f"\n  Fold {fold}  |  "
              f"Train: {df_index[train_idx[0]].date()} - {df_index[train_idx[-1]].date()} ({len(train_idx):,})  |  "
              f"Test:  {df_index[test_idx[0]].date()} - {df_index[test_idx[-1]].date()} ({len(test_idx):,})")
        print(f"         Accuracy: {acc:.3f}   "
              f"Bear: {per_class[-1]:.3f}   "
              f"Side: {per_class[0]:.3f}   "
              f"Bull: {per_class[1]:.3f}")

        all_idx.extend(test_idx.tolist())
        all_pred.extend(y_pred_d.tolist())
        all_act.extend(y_te_d.tolist())
        all_proba.extend(y_proba.tolist())
        all_folds.extend([fold] * len(test_idx))

    return (np.array(all_act), np.array(all_pred), np.array(all_proba),
            all_idx, all_folds, fold_results)


def print_oos_results(actual, predicted, fold_results, target, p2_key):
    """Print combined OOS metrics and fold table. Returns (oos_acc, macro_f1, per_class_recall)."""
    oos_acc  = accuracy_score(actual, predicted)
    macro_f1 = f1_score(actual, predicted, labels=CLASSES, average="macro")
    maj_cls, naive_acc = naive_baseline_accuracy(actual)

    per_class_recall = {
        cls: accuracy_score(actual[actual == cls], predicted[actual == cls])
        if (actual == cls).sum() > 0 else float("nan")
        for cls in CLASSES
    }

    print(f"\n  Overall OOS accuracy  : {oos_acc:.4f}  ({oos_acc*100:.2f}%)")
    print(f"  Naive baseline        : {naive_acc:.4f}  ({naive_acc*100:.2f}%)  "
          f"[always predict {CLASS_NAMES[maj_cls]}]")
    print(f"  Beats naive           : "
          f"{'YES (+{:.2f}pp)'.format((oos_acc-naive_acc)*100) if oos_acc>naive_acc else 'NO ({:.2f}pp below)'.format((naive_acc-oos_acc)*100)}")
    print(f"  Macro F1              : {macro_f1:.4f}")

    print(f"\n  Per-class recall:")
    name_map = {-1: "Bear", 0: "Sideways", 1: "Bull"}
    for cls in CLASSES:
        n   = (actual == cls).sum()
        rec = per_class_recall[cls]
        p2r = P2[p2_key]["recall"][name_map[cls]]
        delta = (rec - p2r) * 100
        print(f"    {CLASS_NAMES[cls]}  n={n:>5}  recall={rec:.4f}  "
              f"({rec*100:.1f}%)  vs P2: {p2r*100:.1f}%  ({delta:+.1f}pp)")

    print(f"\n  Classification report:\n")
    report = classification_report(actual, predicted, labels=CLASSES,
                                   target_names=CLASS_LABELS, digits=3)
    for line in report.split("\n"):
        print(f"    {line}")

    cm = confusion_matrix(actual, predicted, labels=CLASSES)
    print(f"  Confusion matrix:\n")
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

    return oos_acc, macro_f1, per_class_recall


def run_shap(final_model, X, all_features, event_feats, interaction_feats):
    """Run SHAP TreeExplainer and return sorted DataFrame."""
    print("\n  Computing SHAP values ...")
    explainer  = shap.TreeExplainer(final_model)
    shap_raw   = explainer.shap_values(X)
    shap_arr   = np.array(shap_raw)
    if shap_arr.ndim == 3 and shap_arr.shape[0] == len(X):
        sv = [shap_arr[:, :, i] for i in range(shap_arr.shape[2])]
    elif shap_arr.ndim == 3 and shap_arr.shape[0] == 3:
        sv = [shap_arr[i] for i in range(3)]
    else:
        sv = shap_raw

    class_order = ["Bear", "Sideways", "Bull"]
    mean_abs = {cls: np.abs(sv[i]).mean(axis=0) for i, cls in enumerate(class_order)}
    shap_df = pd.DataFrame(mean_abs, index=all_features)
    shap_df["mean_all_classes"] = shap_df.mean(axis=1)
    shap_df["type"] = shap_df.index.map(
        lambda f: "[INTER]" if f in interaction_feats
        else "[EVENT]" if f in event_feats
        else "       "
    )
    return shap_df.sort_values("mean_all_classes", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Load & build interaction features ─────────────────────────────────────
    print("\nLoading data and building interaction features ...")
    raw = pd.read_parquet(DATA_FILE)

    with open(FEAT_FILE) as f:
        feat_meta = json.load(f)
    tech_features = feat_meta["features"]

    # Sentinel fills for partial-coverage event features
    raw["days_since_last_earnings"]      = raw["days_since_last_earnings"].fillna(90)
    raw["days_since_last_product_event"] = raw["days_since_last_product_event"].fillna(180)

    # Build interaction features on the full dataset (global z-score normalisation)
    df = build_interaction_features(raw)

    all_features = tech_features + EVENT_FEATURES + INTERACTION_FEATURES
    print(f"  Tech features        : {len(tech_features)}")
    print(f"  Event features       : {len(EVENT_FEATURES)}")
    print(f"  Interaction features : {len(INTERACTION_FEATURES)}")
    print(f"  Total                : {len(all_features)}")

    # Verify all features present
    missing = [f for f in all_features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # ── Interaction feature summary ────────────────────────────────────────────
    section("INTERACTION FEATURES — Summary Stats")
    for col in INTERACTION_FEATURES:
        s = df[col].dropna()
        print(f"  {col:<35}  mean={s.mean():>8.3f}  std={s.std():>7.3f}  "
              f"range=[{s.min():.2f}, {s.max():.2f}]")

    results = {}

    # ══════════════════════════════════════════════════════════════════════════
    # Model A: dir_1w
    # ══════════════════════════════════════════════════════════════════════════
    section("MODEL A — dir_1w  (57 features, class-balanced weights)")

    target_a = "dir_1w"
    df_a = df[all_features + [target_a]].dropna()
    X_a  = df_a[all_features].values
    y_a  = df_a[target_a].values

    print(f"\n  Samples : {len(df_a):,}  |  "
          f"{df_a.index.min().date()} to {df_a.index.max().date()}")
    counts_a = pd.Series(y_a).value_counts().sort_index()
    for cls in CLASSES:
        n = counts_a.get(cls, 0)
        print(f"  {CLASS_NAMES[cls]}  {n:>5} ({n/len(y_a)*100:.1f}%)")

    act_a, pred_a, proba_a, idx_a, folds_a, folds_res_a = run_walkforward(
        X_a, y_a, df_a.index, "dir_1w"
    )

    section("MODEL A RESULTS — dir_1w")
    acc_a, f1_a, recall_a = print_oos_results(act_a, pred_a, folds_res_a, "dir_1w", "dir_1w")
    results["dir_1w"] = {"acc": acc_a, "f1": f1_a, "recall": recall_a,
                         "act": act_a, "pred": pred_a, "proba": proba_a,
                         "idx": idx_a, "folds": folds_a, "df": df_a, "X": X_a}

    # ══════════════════════════════════════════════════════════════════════════
    # Model B: dir_1m
    # ══════════════════════════════════════════════════════════════════════════
    section("MODEL B — dir_1m  (57 features, class-balanced weights)")

    target_b = "dir_1m"
    df_b = df[all_features + [target_b]].dropna()
    X_b  = df_b[all_features].values
    y_b  = df_b[target_b].values

    print(f"\n  Samples : {len(df_b):,}  |  "
          f"{df_b.index.min().date()} to {df_b.index.max().date()}")
    counts_b = pd.Series(y_b).value_counts().sort_index()
    for cls in CLASSES:
        n = counts_b.get(cls, 0)
        print(f"  {CLASS_NAMES[cls]}  {n:>5} ({n/len(y_b)*100:.1f}%)")

    act_b, pred_b, proba_b, idx_b, folds_b, folds_res_b = run_walkforward(
        X_b, y_b, df_b.index, "dir_1m"
    )

    section("MODEL B RESULTS — dir_1m")
    acc_b, f1_b, recall_b = print_oos_results(act_b, pred_b, folds_res_b, "dir_1m", "dir_1m")
    results["dir_1m"] = {"acc": acc_b, "f1": f1_b, "recall": recall_b,
                         "act": act_b, "pred": pred_b, "proba": proba_b,
                         "idx": idx_b, "folds": folds_b, "df": df_b, "X": X_b}

    # ══════════════════════════════════════════════════════════════════════════
    # Head-to-head comparison
    # ══════════════════════════════════════════════════════════════════════════
    section("HEAD-TO-HEAD COMPARISON")

    name_map = {-1: "Bear", 0: "Sideways", 1: "Bull"}
    print(f"\n  {'Metric':<28}  {'P2 dir_1w':>10}  {'A dir_1w':>10}  {'P2 dir_1m':>10}  {'B dir_1m':>10}")
    print(f"  {'-'*75}")

    def delta_str(new, old):
        d = (new - old) * 100
        return f"({d:+.1f}pp)"

    print(f"  {'OOS Accuracy':<28}  "
          f"{P2['dir_1w']['acc']*100:>9.2f}%  "
          f"{acc_a*100:>8.2f}% {delta_str(acc_a, P2['dir_1w']['acc']):>5}  "
          f"{P2['dir_1m']['acc']*100:>9.2f}%  "
          f"{acc_b*100:>8.2f}% {delta_str(acc_b, P2['dir_1m']['acc']):>5}")
    print(f"  {'Macro F1':<28}  "
          f"{P2['dir_1w']['f1']:>10.3f}  "
          f"{f1_a:>8.3f} ({f1_a-P2['dir_1w']['f1']:+.3f})  "
          f"{P2['dir_1m']['f1']:>10.3f}  "
          f"{f1_b:>8.3f} ({f1_b-P2['dir_1m']['f1']:+.3f})")
    print(f"  {'Naive baseline':<28}  {'37.50%':>10}  {'':>10}  {'52.90%':>10}  {'':>10}")

    for cls in CLASSES:
        name = name_map[cls]
        p2w  = P2["dir_1w"]["recall"][name]
        p2m  = P2["dir_1m"]["recall"][name]
        ra   = recall_a.get(cls, float("nan"))
        rb   = recall_b.get(cls, float("nan"))
        print(f"  {name+' Recall':<28}  "
              f"{p2w*100:>9.1f}%  "
              f"{ra*100:>8.1f}% {delta_str(ra, p2w):>5}  "
              f"{p2m*100:>9.1f}%  "
              f"{rb*100:>8.1f}% {delta_str(rb, p2m):>5}")

    # Determine winner by macro F1 (most balanced metric for 3-class imbalanced problem)
    winner_key = "dir_1w" if f1_a >= f1_b else "dir_1m"
    winner_res  = results[winner_key]
    winner_p2   = P2[winner_key]
    winner_f1   = f1_a if winner_key == "dir_1w" else f1_b
    winner_acc  = acc_a if winner_key == "dir_1w" else acc_b
    winner_name = f"Model {'A' if winner_key=='dir_1w' else 'B'} ({winner_key})"

    print(f"\n  WINNER by Macro F1 : {winner_name}  "
          f"(F1={winner_f1:.4f}  vs P2 {winner_p2['f1']:.4f}  "
          f"delta={winner_f1-winner_p2['f1']:+.3f})")
    if winner_f1 > winner_p2["f1"]:
        print(f"  >>> Interaction features IMPROVED Macro F1 vs Phase 2 baseline <<<")
    else:
        print(f"  >>> Interaction features did NOT improve Macro F1 vs Phase 2 baseline <<<")

    # ══════════════════════════════════════════════════════════════════════════
    # Save outputs for winner
    # ══════════════════════════════════════════════════════════════════════════
    section(f"SAVING OUTPUTS  ({winner_name})")

    win = results[winner_key]
    pred_df = pd.DataFrame({
        "actual"       : win["act"],
        "predicted"    : win["pred"],
        "correct"      : (win["act"] == win["pred"]).astype(int),
        "prob_bear"    : win["proba"][:, 0],
        "prob_sideways": win["proba"][:, 1],
        "prob_bull"    : win["proba"][:, 2],
        "confidence"   : win["proba"].max(axis=1),
        "fold"         : win["folds"],
        "target"       : winner_key,
    }, index=win["df"].index[win["idx"]])
    pred_df.index.name = "date"
    pred_df = pred_df.sort_index()

    PRED_FILE.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(PRED_FILE)
    print(f"\n  Predictions saved : {PRED_FILE.name}  ({len(pred_df):,} rows)")

    # Train final model on full winner dataset
    print(f"  Training final model on full {winner_key} dataset ...")
    y_enc_final = np.array([LABEL_ENCODE[v] for v in win["df"][winner_key].values])
    y_orig_final = np.array([LABEL_DECODE[v] for v in y_enc_final])
    final_weights = compute_sample_weight(class_weight="balanced", y=y_orig_final)
    final_model = XGBClassifier(**XGB_PARAMS)
    final_model.fit(win["X"], y_enc_final, sample_weight=final_weights)

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(final_model, f)
    print(f"  Model saved       : {MODEL_FILE.name}")

    # ══════════════════════════════════════════════════════════════════════════
    # SHAP on winner
    # ══════════════════════════════════════════════════════════════════════════
    section(f"SHAP ANALYSIS — {winner_name}")

    shap_df = run_shap(final_model, win["X"], all_features, EVENT_FEATURES, INTERACTION_FEATURES)

    print(f"\n  Top 25 features by mean |SHAP|:\n")
    print(f"  {'Rank':>4}  {'Feature':<32}  {'Bear':>7}  {'Side':>7}  {'Bull':>7}  {'Mean':>7}  Type")
    print(f"  {'-'*80}")
    for rank, (feat, row) in enumerate(shap_df.head(25).iterrows(), 1):
        print(f"  {rank:>4}  {feat:<32}  "
              f"{row['Bear']:>7.4f}  {row['Sideways']:>7.4f}  "
              f"{row['Bull']:>7.4f}  {row['mean_all_classes']:>7.4f}  {row['type']}")

    # Interaction feature SHAP breakdown
    inter_shap = shap_df[shap_df["type"] == "[INTER]"].copy()
    event_shap = shap_df[shap_df["type"] == "[EVENT]"].copy()

    print(f"\n  Interaction features ranked by mean |SHAP|:")
    for rank, (feat, row) in enumerate(inter_shap.iterrows(), 1):
        dominant = max(["Bear","Sideways","Bull"], key=lambda c: row[c])
        shap_rank = shap_df.index.get_loc(feat) + 1
        print(f"    {rank}. {feat:<35}  mean={row['mean_all_classes']:.4f}  "
              f"(overall rank #{shap_rank}, strongest for {dominant})")

    print(f"\n  Top event features ranked by mean |SHAP|  (top 8):")
    for rank, (feat, row) in enumerate(event_shap.head(8).iterrows(), 1):
        dominant = max(["Bear","Sideways","Bull"], key=lambda c: row[c])
        shap_rank = shap_df.index.get_loc(feat) + 1
        print(f"    {rank}. {feat:<35}  mean={row['mean_all_classes']:.4f}  "
              f"(overall rank #{shap_rank}, strongest for {dominant})")

    SHAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    shap_df.to_csv(SHAP_FILE)
    print(f"\n  SHAP summary saved : {SHAP_FILE.name}")

    # ══════════════════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════════════════
    section("PHASE 3 STEP 4 SUMMARY")

    p2_champ_f1  = P2["dir_1w"]["f1"]   # phase 2 champion
    overall_best = "Phase 2 (dir_1w, 38.35%)" if winner_f1 < p2_champ_f1 else winner_name
    p2_str = f"F1={p2_champ_f1:.3f}"
    w_str  = f"F1={winner_f1:.3f}"
    name_map = {-1: "Bear", 0: "Sideways", 1: "Bull"}

    print(f"""
  Model A (dir_1w, 57 feat) : acc={acc_a:.4f}  F1={f1_a:.4f}  vs P2: {P2['dir_1w']['f1']:.4f} ({f1_a-P2['dir_1w']['f1']:+.3f})
  Model B (dir_1m, 57 feat) : acc={acc_b:.4f}  F1={f1_b:.4f}  vs P2: {P2['dir_1m']['f1']:.4f} ({f1_b-P2['dir_1m']['f1']:+.3f})

  Winner (by Macro F1)      : {winner_name}  ({w_str})
  Phase 2 champion          : dir_1w weighted  ({p2_str})
  Overall best model        : {overall_best}

  Interaction feature SHAP ranks:
    {chr(10).join(f"    {f}: rank #{shap_df.index.get_loc(f)+1}  mean_shap={shap_df.loc[f,'mean_all_classes']:.4f}" for f in INTERACTION_FEATURES if f in shap_df.index)}

  Top event feature by SHAP : {event_shap.index[0] if len(event_shap) > 0 else 'n/a'}
  Top overall feature (SHAP): {shap_df.index[0]}

  Saved:
    {MODEL_FILE}
    {PRED_FILE}
    {SHAP_FILE}
""")
    print("Phase 3 Step 4 complete.\n")
