# ML Standards -- aapl_ml

## Walk-forward validation (non-negotiable)
Use TimeSeriesSplit, not random k-fold. Split on date order.

    tscv = TimeSeriesSplit(n_splits=5)
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        pass  # train rows always earlier than test rows

WRONG: KFold(shuffle=True)  -- destroys time ordering

## Holdout
- >= 2024-01-01 (to 2025-03-19)
- Never touched during training, walk-forward, or hyperparameter tuning
- Evaluated once, after all model decisions are made

## No lookahead bias
Features at date t must only use data available at close of day t.

    # BANNED
    df[chr(34) + chr(34).join(['feat']) + chr(34)] = df[chr(34)close chr(34)].shift(-1)
    # CORRECT
    df["feat"] = df["close"].pct_change()
    df["ma"] = df["close"].rolling(20).mean()  # past 20 only

## Baseline comparison
Every new model vs:
1. Naive baseline: always Bull (52.90% accuracy, F1=0.230)
2. Current champion: xgb_phase3_champion.pkl (F1=0.375)

Report delta explicitly: +0.008 F1, not just 0.383.

## Champion thresholds
- F1 > 0.375 on OOS to qualify as new champion
- Sideways recall > 30% (don't collapse the middle class)
- Bear recall > 20%

## SHAP
Run after every new champion. Save shap_summary*.csv to models/.

