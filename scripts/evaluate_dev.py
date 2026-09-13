import pandas as pd

from src.config import PROCESSED_DIR, PROJECT_ROOT, HORIZONS_MIN, RANDOM_SEED
from src.evaluation import walk_forward_splits, mae, skill_score, persistence_baseline, linear_baseline
from src.models import train_xgboost, predict_xgboost, train_lstm, predict_lstm, TIERS

REPORTS_DIR = PROJECT_ROOT / 'reports'
N_SPLITS = 5

# stamped into the output filenames so a reseeded run does not overwrite the first one
RUN_TAG = f'seed{RANDOM_SEED}'

# parquet keeps dtypes and the timestamp timezones
df = pd.read_parquet(PROCESSED_DIR / 'development.parquet')
print('development rows', len(df))
print('period counts\n', df['period'].value_counts())
print()

# ---------- RQ1: encoded domain knowledge vs learned representation ----------
# One pass per horizon per split. Both models and both baselines are predicted on the same test block, 
# then all four are cut down to the rows they share before anything is measured.

rq1_rows = []

for horizon in HORIZONS_MIN:
    target_col = f'target_glucose_{horizon}'
    truth = df[target_col]

    # splits are rebuilt per horizon
    for i, (train_idx, test_idx) in enumerate(walk_forward_splits(len(df), N_SPLITS, horizon)):
        test = df.iloc[test_idx]

        # both fit on training rows only
        xgb = train_xgboost(df, train_idx, target_col)
        lstm, scaler = train_lstm(df, train_idx, target_col)

        preds = {
            'persistence': persistence_baseline(test, horizon),
            'linear': linear_baseline(test, horizon),
            'xgboost': predict_xgboost(df, xgb, test_idx),
            'lstm': predict_lstm(df, lstm, scaler, test_idx, target_col),
        }

        # XGBoost drops rows with a missing feature, the LSTM only produces a value at the end of a complete 60-step window -> score them all on the rows they share
        common = truth.iloc[test_idx].dropna().index
        for pred in preds.values():
            common = common.intersection(pred.dropna().index)

        if len(common) == 0:
            print(f'horizon {horizon} split {i}: no scoreable rows, skipped')
            continue

        truth_common = truth.loc[common]

        # the baseline is measured on the same rows as the models it scores
        base_mae = mae(truth_common, preds['persistence'].loc[common])

        row = {'horizon': horizon, 'split': i, 'n_scored': len(common)}
        for name, pred in preds.items():
            model_mae = mae(truth_common, pred.loc[common])
            row[f'{name}_mae'] = round(model_mae, 3)
            row[f'{name}_skill'] = round(skill_score(model_mae, base_mae), 3)

        rq1_rows.append(row)
        print(f'horizon {horizon} split {i}  n={len(common)}  '
              f'xgb {row["xgboost_mae"]}  lstm {row["lstm_mae"]}  base {row["persistence_mae"]}')

rq1 = pd.DataFrame(rq1_rows)
rq1.to_csv(REPORTS_DIR / f'dev_rq1_{RUN_TAG}.csv', index=False)

# ---------- RQ1 summary ----------
# Four horizons averaged over five splits, plus the paired win count, which is the comparison that actually answers RQ1

print()
print('mean MAE by horizon')
print(rq1.groupby('horizon')[['persistence_mae', 'linear_mae', 'xgboost_mae', 'lstm_mae']].mean().round(3))
print()
print('mean skill by horizon')
print(rq1.groupby('horizon')[['xgboost_skill', 'lstm_skill']].mean().round(3))
print()

# RQ1 is a paired question -- how often does XGBoost win, not whose average is lower
rq1['xgb_wins'] = rq1['xgboost_mae'] < rq1['lstm_mae']
print('XGBoost wins, out of', N_SPLITS, 'splits')
print(rq1.groupby('horizon')['xgb_wins'].sum())
print()
print('rows scored per split')
print(rq1.groupby('horizon')['n_scored'].agg(['min', 'mean', 'max']).round(0))
print()

# ---------- RQ2: feature group contribution ----------
# The same XGBoost trained six times per split, each tier adding one feature group to the one before it
abl_rows = []

for horizon in HORIZONS_MIN:
    target_col = f'target_glucose_{horizon}'
    truth = df[target_col]

    for i, (train_idx, test_idx) in enumerate(walk_forward_splits(len(df), N_SPLITS, horizon)):
        test = df.iloc[test_idx]

        preds = {}
        for tier_name, features in TIERS.items():
            # features has to be passed to both calls or predict sees columns the model was never fit on
            model = train_xgboost(df, train_idx, target_col, features=features)
            preds[tier_name] = predict_xgboost(df, model, test_idx, features=features)

        # a tier with more features drops more rows, so without this the ladder would partly measure which rows survived instead of which features helped
        common = truth.iloc[test_idx].dropna().index
        for pred in preds.values():
            common = common.intersection(pred.dropna().index)

        if len(common) == 0:
            continue

        truth_common = truth.loc[common]
        # one baseline for all six tiers
        base_mae = mae(truth_common, persistence_baseline(test, horizon).loc[common])

        for tier_name, pred in preds.items():
            tier_mae = mae(truth_common, pred.loc[common])
            abl_rows.append({
                'horizon': horizon,
                'split': i,
                'tier': tier_name,
                'mae': round(tier_mae, 3),
                'skill': round(skill_score(tier_mae, base_mae), 3),
                'n_scored': len(common),
            })

    print('ablation done for horizon', horizon)

abl = pd.DataFrame(abl_rows)
abl.to_csv(REPORTS_DIR / f'dev_ablation_{RUN_TAG}.csv', index=False)


# ---------- RQ2 summary ----------
# six tiers by four horizons

table = abl.groupby(['tier', 'horizon'])['skill'].mean().unstack()
table = table.reindex(TIERS.keys())   # groupby sorts alphabetically, restore the ladder

print()
print('mean skill by tier')
print(table.round(3))
print()
print('marginal gain from each tier')
print(table.diff().round(3))   # each row minus the one above it
print()
print('written to', REPORTS_DIR)