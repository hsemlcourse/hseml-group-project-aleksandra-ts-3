import pytest

from src.config import DATA_RAW
from src.preprocessing import add_engineered_features, clean_dataframe, full_pipeline, load_raw


@pytest.mark.skipif(not DATA_RAW.is_file(), reason='Download data: python -m src.scripts.download_data')
def test_load_and_clean():
    df = load_raw()
    assert df.shape[0] >= 1400
    clean, log = clean_dataframe(df)
    assert 'n_rows_in' in log
    assert 'Attrition' in clean.columns
    assert set(clean['Attrition'].unique()).issubset({0, 1})


@pytest.mark.skipif(not DATA_RAW.is_file(), reason='нет data/raw/*.csv')
def test_split_and_encode_shapes():
    enc = full_pipeline(with_engineering=True)
    total_rows = enc.X_train.shape[0] + enc.X_val.shape[0] + enc.X_test.shape[0]
    assert total_rows == 1470
    assert enc.y_train.sum() > 0 and enc.y_train.sum() < len(enc.y_train)


@pytest.mark.skipif(not DATA_RAW.is_file(), reason='нет data/raw/*.csv')
def test_baseline_fewer_features_than_engineered():
    baseline = full_pipeline(with_engineering=False)
    engineered = full_pipeline(with_engineering=True)
    assert baseline.X_train.shape[1] < engineered.X_train.shape[1]


def test_engineered_columns_present():
    if not DATA_RAW.is_file():
        pytest.skip('нет данных')
    df = load_raw()
    clean_df, _ = clean_dataframe(df)
    fe_df = add_engineered_features(clean_df)
    for i in ('IncomeToRoleAvg', 'TenureRatio', 'SatisfactionIndex', 'HighRiskFlag'):
        assert i in fe_df.columns
