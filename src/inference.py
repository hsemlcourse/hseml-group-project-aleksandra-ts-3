import json

import joblib
import pandas as pd

from src.config import PROJECT_ROOT
from src.deploy_preprocessing import transform_rows

BUNDLE_PATH = PROJECT_ROOT / 'models' / 'deploy_bundle.joblib'
META_PATH = PROJECT_ROOT / 'models' / 'deploy_meta.json'
DEFAULTS_PATH = PROJECT_ROOT / 'models' / 'default_employee.json'

RISK_THRESHOLDS = (0.35, 0.55)


def risk_label(probability):
    if probability < RISK_THRESHOLDS[0]:
        return 'низкий'
    if probability < RISK_THRESHOLDS[1]:
        return 'средний'
    return 'высокий'


class AttritionPredictor:
    def __init__(self, bundle_path=None):
        path = bundle_path or BUNDLE_PATH
        if not path.is_file():
            raise FileNotFoundError(
                f'Артефакт не найден: {path}'
            )
        payload = joblib.load(path)
        self.model = payload['model']
        self.preprocessor = payload['preprocessor']
        self.feature_columns = payload['feature_columns']
        self.meta = {}
        if META_PATH.is_file():
            self.meta = json.loads(META_PATH.read_text(encoding='utf-8'))

    def predict_proba(self, employees):
        matrix = transform_rows(employees, self.preprocessor)
        return self.model.predict_proba(matrix)[:, 1]

    def predict(self, employees, threshold=0.5):
        return (self.predict_proba(employees) >= threshold).astype(int)

    def explain_one(self, employee):
        frame = pd.DataFrame([employee])
        prob = float(self.predict_proba(frame)[0])
        pred = int(prob >= 0.5)
        return {
            'attrition_probability': round(prob, 4),
            'attrition_predicted': pred,
            'risk_level': risk_label(prob),
            'model': self.meta.get('model_name', 'RandomForest_tuned'),
        }


def load_default_employee():
    if DEFAULTS_PATH.is_file():
        return json.loads(DEFAULTS_PATH.read_text(encoding='utf-8'))
    return {}
