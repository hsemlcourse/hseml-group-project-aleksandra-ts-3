from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import DATA_PROCESSED, DATA_RAW, DROP_ALWAYS, RANDOM_STATE, TARGET_COL

ORDINAL_ORDERS = {
    'Education': [1, 2, 3, 4, 5],
    'EnvironmentSatisfaction': [1, 2, 3, 4],
    'JobInvolvement': [1, 2, 3, 4],
    'JobSatisfaction': [1, 2, 3, 4],
    'PerformanceRating': [1, 2, 3, 4],
    'RelationshipSatisfaction': [1, 2, 3, 4],
    'WorkLifeBalance': [1, 2, 3, 4],
    'JobLevel': [1, 2, 3, 4, 5],
    'StockOptionLevel': [0, 1, 2, 3],
}

NOMINAL_COLS = [
    'BusinessTravel',
    'Department',
    'EducationField',
    'Gender',
    'JobRole',
    'MaritalStatus',
    'OverTime',
]

NUMERIC_COLS_BASE = [
    'Age',
    'DailyRate',
    'DistanceFromHome',
    'HourlyRate',
    'MonthlyIncome',
    'MonthlyRate',
    'NumCompaniesWorked',
    'PercentSalaryHike',
    'TotalWorkingYears',
    'TrainingTimesLastYear',
    'YearsAtCompany',
    'YearsInCurrentRole',
    'YearsSinceLastPromotion',
    'YearsWithCurrManager',
]


def _binary_target(series):
    return series.map({'Yes': 1, 'No': 0}).astype(int)


def load_raw(path=None):
    source = path or str(DATA_RAW)
    return pd.read_csv(source)


def clean_dataframe(df):
    log = {}
    log['n_rows_in'] = len(df)

    drop_cols = [i for i in DROP_ALWAYS if i in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        log['dropped_constant_or_id'] = len(drop_cols)

    log['n_duplicates_dropped'] = int(df.duplicated().sum())
    df = df.drop_duplicates().reset_index(drop=True)
    log['missing_cells'] = int(df.isna().sum().sum())

    for i in df.columns:
        if i == TARGET_COL:
            continue
        known_col = i in NUMERIC_COLS_BASE or i in ORDINAL_ORDERS or i in NOMINAL_COLS
        if not known_col or not df[i].isna().any():
            continue
        if i in NUMERIC_COLS_BASE:
            df[i] = df[i].fillna(df[i].median())
        else:
            df[i] = df[i].fillna(df[i].mode().iloc[0])

    df[TARGET_COL] = _binary_target(df[TARGET_COL])
    for i in NUMERIC_COLS_BASE:
        if i in df.columns:
            df[i] = pd.to_numeric(df[i], errors='coerce')

    before_rows = len(df)
    for i in NUMERIC_COLS_BASE:
        if i not in df.columns:
            continue
        q1, q3 = df[i].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_outliers = int(((df[i] < low) | (df[i] > high)).sum())
        if 0 < n_outliers < 0.05 * len(df):
            df[i] = df[i].clip(lower=low, upper=high)

    log['rows_after_winsor'] = len(df)
    if len(df) < before_rows:
        log['note'] = 'rows_unchanged_or_clipped'
    return df, log


def add_engineered_features(df):
    out = df.copy()
    role_avg = out.groupby('JobRole')['MonthlyIncome'].transform('mean')
    out['IncomeToRoleAvg'] = out['MonthlyIncome'] / (role_avg.replace(0, np.nan) + 1e-6)
    out['TenureRatio'] = out['YearsAtCompany'] / (out['Age'].replace(0, np.nan) + 1e-6)

    sat_cols = [i for i in ('JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance') if i in out.columns]
    if sat_cols:
        out['SatisfactionIndex'] = out[sat_cols].mean(axis=1)

    overtime = (out['OverTime'] == 'Yes').astype(int)
    low_job_sat = (out['JobSatisfaction'] <= 2).astype(int) if 'JobSatisfaction' in out.columns else 0
    low_tenure = (out['YearsAtCompany'] < 3).astype(int)
    out['HighRiskFlag'] = overtime + low_job_sat + low_tenure
    return out


@dataclass
class EncodedData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list
    ohe: object
    ordinal: object
    scaler: object


def _get_feature_columns(df, *, with_engineering):
    num_cols = [i for i in NUMERIC_COLS_BASE if i in df.columns]
    engineered = ('IncomeToRoleAvg', 'TenureRatio', 'SatisfactionIndex', 'HighRiskFlag')
    if with_engineering:
        extra_num = [i for i in engineered if i in df.columns and np.issubdtype(df[i].dtype, np.number)]
        num_cols = list(dict.fromkeys(num_cols + extra_num))

    ord_cols = [i for i in ORDINAL_ORDERS if i in df.columns]
    nom_cols = [i for i in NOMINAL_COLS if i in df.columns]
    use_cols = [i for i in (num_cols + ord_cols + nom_cols) if i in df.columns]

    X = df[use_cols].copy()
    y = df[TARGET_COL].to_numpy()
    return X, y, num_cols, ord_cols, nom_cols


def build_matrices(df, *, with_engineering=True):
    X_df, y, num_cols, ord_cols, nom_cols = _get_feature_columns(df, with_engineering=with_engineering)

    idx = X_df.index
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, y, test_size=0.36, stratify=y, random_state=RANDOM_STATE
    )
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=0.55555556, stratify=y_temp, random_state=RANDOM_STATE
    )

    train = X_df.loc[train_idx].copy()
    val = X_df.loc[val_idx].copy()
    test = X_df.loc[test_idx].copy()

    ordinal_enc = None
    if ord_cols:
        for i in ord_cols:
            train[i] = train[i].astype(int)
            val[i] = val[i].astype(int)
            test[i] = test[i].astype(int)

        cats = [ORDINAL_ORDERS[i] for i in ord_cols]
        ordinal_enc = OrdinalEncoder(categories=cats, handle_unknown='use_encoded_value', unknown_value=-1)
        ordinal_enc.fit(train[ord_cols])
        for i in (train, val, test):
            i[ord_cols] = ordinal_enc.transform(i[ord_cols])

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(train[nom_cols])
    nom_train = ohe.transform(train[nom_cols])
    nom_val = ohe.transform(val[nom_cols])
    nom_test = ohe.transform(test[nom_cols])
    nom_names = list(ohe.get_feature_names_out(nom_cols))

    base_train = train[num_cols + ord_cols].to_numpy().astype(float)
    base_val = val[num_cols + ord_cols].to_numpy().astype(float)
    base_test = test[num_cols + ord_cols].to_numpy().astype(float)

    X_train = np.hstack([base_train, nom_train]) if base_train.size else nom_train
    X_val = np.hstack([base_val, nom_val]) if base_val.size else nom_val
    X_test = np.hstack([base_test, nom_test]) if base_test.size else nom_test

    scaler = StandardScaler()
    if num_cols or ord_cols:
        n_ord = len(num_cols) + len(ord_cols)
        scaler.fit(X_train[:, :n_ord])
        for i in (X_train, X_val, X_test):
            i[:, :n_ord] = scaler.transform(i[:, :n_ord])

    feature_names = num_cols + [f'ord_{col}' for col in ord_cols] + nom_names
    return EncodedData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        ohe=ohe,
        ordinal=ordinal_enc,
        scaler=scaler,
    )


def full_pipeline(path=None, *, with_engineering=True, save_dir=None):
    df, _ = clean_dataframe(load_raw(path))
    if with_engineering:
        df = add_engineered_features(df)

    out = build_matrices(df, with_engineering=with_engineering)
    if save_dir is not None:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out_dir = Path(save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / 'feature_names.txt').write_text('\n'.join(out.feature_names), encoding='utf-8')
    return out
