import json

import pandas as pd
import pytest

from src.config import DATA_RAW, PROJECT_ROOT
from src.deploy_preprocessing import fit_preprocessor, prepare_training_frame, transform_rows
from src.inference import AttritionPredictor, risk_label

BUNDLE = PROJECT_ROOT / 'models' / 'deploy_bundle.joblib'


@pytest.mark.skipif(not DATA_RAW.is_file(), reason='python -m src.scripts.download_data')
def test_fit_transform_same_width():
    df = prepare_training_frame()
    state, x_train, _ = fit_preprocessor(df)
    assert x_train.shape[1] == len(state.feature_names)
    sample = df.drop(columns=['Attrition']).iloc[:3]
    x_batch = transform_rows(sample, state)
    assert x_batch.shape == (3, len(state.feature_names))


def test_risk_label_buckets():
    assert risk_label(0.1) == 'низкий'
    assert risk_label(0.4) == 'средний'
    assert risk_label(0.8) == 'высокий'


@pytest.mark.skipif(not BUNDLE.is_file(), reason='python -m src.scripts.export_deploy')
def test_predictor_bundle():
    defaults_path = PROJECT_ROOT / 'models' / 'default_employee.json'
    assert defaults_path.is_file()
    defaults = json.loads(defaults_path.read_text(encoding='utf-8'))
    predictor = AttritionPredictor()
    out = predictor.explain_one(defaults)
    assert 0.0 <= out['attrition_probability'] <= 1.0
    assert out['risk_level'] in {'низкий', 'средний', 'высокий'}
    batch = pd.DataFrame([defaults, defaults])
    probs = predictor.predict_proba(batch)
    assert probs.shape == (2,)
