import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import RANDOM_STATE, TARGET_COL
from src.modeling import classification_metrics, make_baseline_logistic
from src.preprocessing import (
    NOMINAL_COLS,
    NUMERIC_COLS_BASE,
    ORDINAL_ORDERS,
    add_engineered_features,
    clean_dataframe,
    load_raw,
)


def split_raw(df):
    idx = df.index
    y = df[TARGET_COL].to_numpy()
    train_idx, temp_idx, _, y_temp = train_test_split(
        idx, y, test_size=0.36, stratify=y, random_state=RANDOM_STATE
    )
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=0.55555556, stratify=y_temp, random_state=RANDOM_STATE
    )
    return df.loc[train_idx].copy(), df.loc[val_idx].copy(), df.loc[test_idx].copy()


def synth_expand_train(train_df, num_cols, ord_cols, target_rows=10000, noise_scale=0.03):
    out = train_df.copy()
    while len(out) < target_rows:
        need = min(target_rows - len(out), len(train_df))
        sampled = train_df.sample(n=need, replace=True, random_state=np.random.randint(0, 10**9)).copy()

        for col in num_cols:
            std = float(train_df[col].std()) if pd.notna(train_df[col].std()) else 0.0
            if std > 0:
                sampled[col] = sampled[col] + np.random.normal(0, std * noise_scale, size=len(sampled))

        for col in ord_cols:
            sampled[col] = sampled[col].round().clip(min(ORDINAL_ORDERS[col]), max(ORDINAL_ORDERS[col]))

        int_like = {
            'Age',
            'DistanceFromHome',
            'NumCompaniesWorked',
            'PercentSalaryHike',
            'TotalWorkingYears',
            'TrainingTimesLastYear',
            'YearsAtCompany',
            'YearsInCurrentRole',
            'YearsSinceLastPromotion',
            'YearsWithCurrManager',
            'JobLevel',
            'StockOptionLevel',
        }
        for col in int_like:
            if col in sampled.columns:
                sampled[col] = np.round(sampled[col]).astype(int)

        out = pd.concat([out, sampled], ignore_index=True)
    return out.iloc[:target_rows].copy()


def encode_fit_transform(train_df, val_df, test_df, num_cols, ord_cols, nom_cols):
    for col in ord_cols:
        train_df[col] = train_df[col].astype(int)
        val_df[col] = val_df[col].astype(int)
        test_df[col] = test_df[col].astype(int)

    ord_enc = OrdinalEncoder(
        categories=[ORDINAL_ORDERS[col] for col in ord_cols],
        handle_unknown='use_encoded_value',
        unknown_value=-1,
    )
    ord_enc.fit(train_df[ord_cols])
    train_df[ord_cols] = ord_enc.transform(train_df[ord_cols])
    val_df[ord_cols] = ord_enc.transform(val_df[ord_cols])
    test_df[ord_cols] = ord_enc.transform(test_df[ord_cols])

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(train_df[nom_cols])
    tr_nom = ohe.transform(train_df[nom_cols])
    va_nom = ohe.transform(val_df[nom_cols])
    te_nom = ohe.transform(test_df[nom_cols])

    tr_base = train_df[num_cols + ord_cols].to_numpy().astype(float)
    va_base = val_df[num_cols + ord_cols].to_numpy().astype(float)
    te_base = test_df[num_cols + ord_cols].to_numpy().astype(float)

    x_train = np.hstack([tr_base, tr_nom])
    x_val = np.hstack([va_base, va_nom])
    x_test = np.hstack([te_base, te_nom])

    scaler = StandardScaler()
    n_ord = len(num_cols) + len(ord_cols)
    scaler.fit(x_train[:, :n_ord])
    x_train[:, :n_ord] = scaler.transform(x_train[:, :n_ord])
    x_val[:, :n_ord] = scaler.transform(x_val[:, :n_ord])
    x_test[:, :n_ord] = scaler.transform(x_test[:, :n_ord])
    return x_train, x_val, x_test


def main():
    np.random.seed(RANDOM_STATE)
    df_raw = load_raw()
    df_clean, _ = clean_dataframe(df_raw)
    df = add_engineered_features(df_clean)

    num_cols = [col for col in NUMERIC_COLS_BASE if col in df.columns]
    engineered = ['IncomeToRoleAvg', 'TenureRatio', 'SatisfactionIndex', 'HighRiskFlag']
    num_cols = list(dict.fromkeys(num_cols + [col for col in engineered if col in df.columns]))
    ord_cols = [col for col in ORDINAL_ORDERS if col in df.columns]
    nom_cols = [col for col in NOMINAL_COLS if col in df.columns]

    train, val, test = split_raw(df)
    train_aug = synth_expand_train(train, num_cols, ord_cols, target_rows=10000, noise_scale=0.03)

    x_tr, x_va, x_te = encode_fit_transform(train.copy(), val.copy(), test.copy(), num_cols, ord_cols, nom_cols)
    y_tr = train[TARGET_COL].to_numpy()
    y_va = val[TARGET_COL].to_numpy()
    y_te = test[TARGET_COL].to_numpy()

    x_tr_aug, x_va_aug, x_te_aug = encode_fit_transform(
        train_aug.copy(),
        val.copy(),
        test.copy(),
        num_cols,
        ord_cols,
        nom_cols,
    )
    y_tr_aug = train_aug[TARGET_COL].to_numpy()

    model = make_baseline_logistic()
    model.fit(x_tr, y_tr)
    orig_val = classification_metrics(y_va, model.predict(x_va), model.predict_proba(x_va)[:, 1])
    orig_test = classification_metrics(y_te, model.predict(x_te), model.predict_proba(x_te)[:, 1])

    model_aug = make_baseline_logistic()
    model_aug.fit(x_tr_aug, y_tr_aug)
    aug_val = classification_metrics(y_va, model_aug.predict(x_va_aug), model_aug.predict_proba(x_va_aug)[:, 1])
    aug_test = classification_metrics(y_te, model_aug.predict(x_te_aug), model_aug.predict_proba(x_te_aug)[:, 1])

    out_dir = Path('data/processed')
    out_dir.mkdir(parents=True, exist_ok=True)
    train_aug.to_csv(out_dir / 'train_synthetic_10000.csv', index=False)

    metrics = {
        'train_original_rows': int(len(train)),
        'train_synthetic_rows': int(len(train_aug)),
        'original_val': orig_val,
        'original_test': orig_test,
        'synthetic_val': aug_val,
        'synthetic_test': aug_test,
    }
    (out_dir / 'synthetic_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
