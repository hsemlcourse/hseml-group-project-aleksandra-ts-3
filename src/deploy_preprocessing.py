import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src.config import DROP_ALWAYS, RANDOM_STATE, TARGET_COL
from src.preprocessing import (
    NUMERIC_COLS_BASE,
    ORDINAL_ORDERS,
    _get_feature_columns,
    add_engineered_features,
    clean_dataframe,
    load_raw,
)

FINAL_RF_PARAMS = {
    'n_estimators': 700,
    'max_depth': 6,
    'min_samples_leaf': 8,
    'min_samples_split': 10,
    'class_weight': 'balanced_subsample',
}


class PreprocessorState:
    def __init__(self, num_cols, ord_cols, nom_cols, ordinal, ohe, scaler, feature_names, n_scaled):
        self.num_cols = num_cols
        self.ord_cols = ord_cols
        self.nom_cols = nom_cols
        self.ordinal = ordinal
        self.ohe = ohe
        self.scaler = scaler
        self.feature_names = feature_names
        self.n_scaled = n_scaled


def enrich_for_inference(df):
    work = df.copy()
    drop = [i for i in DROP_ALWAYS if i in work.columns]
    if drop:
        work = work.drop(columns=drop)
    if TARGET_COL in work.columns:
        work = work.drop(columns=[TARGET_COL])
    if 'IncomeToRoleAvg' not in work.columns:
        work = add_engineered_features(work)
    return work


def _transform_frame(other, state):
    ord_cols = state.ord_cols
    nom_cols = state.nom_cols
    num_cols = state.num_cols

    if ord_cols and state.ordinal is not None:
        for i in ord_cols:
            other[i] = other[i].astype(int)
        other[ord_cols] = state.ordinal.transform(other[ord_cols])

    nom_other = state.ohe.transform(other[nom_cols])
    base_other = other[num_cols + ord_cols].to_numpy().astype(float)
    x_other = np.hstack([base_other, nom_other]) if base_other.size else nom_other

    if state.n_scaled:
        x_other[:, : state.n_scaled] = state.scaler.transform(x_other[:, : state.n_scaled])
    return x_other


def fit_preprocessor(df, with_engineering=True):
    x_df, y, num_cols, ord_cols, nom_cols = _get_feature_columns(df, with_engineering=with_engineering)

    train_idx, _, y_train, _ = train_test_split(
        x_df.index,
        y,
        test_size=0.36,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    train = x_df.loc[train_idx].copy()

    ordinal_enc = None
    if ord_cols:
        for i in ord_cols:
            train[i] = train[i].astype(int)
        cats = [ORDINAL_ORDERS[i] for i in ord_cols]
        ordinal_enc = OrdinalEncoder(categories=cats, handle_unknown='use_encoded_value', unknown_value=-1)
        ordinal_enc.fit(train[ord_cols])

    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(train[nom_cols])

    nom_train = ohe.transform(train[nom_cols])
    base_train = train[num_cols + ord_cols].to_numpy().astype(float)
    x_train = np.hstack([base_train, nom_train]) if base_train.size else nom_train

    scaler = StandardScaler()
    n_scaled = len(num_cols) + len(ord_cols)
    if n_scaled:
        scaler.fit(x_train[:, :n_scaled])
        x_train[:, :n_scaled] = scaler.transform(x_train[:, :n_scaled])

    feature_names = num_cols + [f'ord_{i}' for i in ord_cols] + list(ohe.get_feature_names_out(nom_cols))
    state = PreprocessorState(
        num_cols,
        ord_cols,
        nom_cols,
        ordinal_enc,
        ohe,
        scaler,
        feature_names,
        n_scaled,
    )
    return state, x_train, y_train


def transform_rows(df, state):
    work = enrich_for_inference(df)

    missing = [i for i in (state.num_cols + state.ord_cols + state.nom_cols) if i not in work.columns]
    if missing:
        raise ValueError(f'Не хватает полей: {missing}')

    return _transform_frame(work, state)


def prepare_training_frame(path=None, with_engineering=True):
    raw = load_raw(path)
    clean, _ = clean_dataframe(raw)
    if with_engineering:
        clean = add_engineered_features(clean)
    return clean


def default_employee_row(df):
    fe_cols = {'IncomeToRoleAvg', 'TenureRatio', 'SatisfactionIndex', 'HighRiskFlag'}
    row = {}
    for i in df.columns:
        if i == TARGET_COL or i in fe_cols:
            continue
        series = df[i]
        if i in NUMERIC_COLS_BASE:
            row[i] = float(series.median())
        elif i in ORDINAL_ORDERS:
            row[i] = int(series.mode().iloc[0])
        else:
            row[i] = series.mode().iloc[0]
    return row
