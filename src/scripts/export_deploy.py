import json

import joblib
from sklearn.ensemble import RandomForestClassifier

from src.config import PROJECT_ROOT, RANDOM_STATE
from src.deploy_preprocessing import (
    FINAL_RF_PARAMS,
    default_employee_row,
    fit_preprocessor,
    prepare_training_frame,
)
from src.modeling import classification_metrics

BUNDLE_PATH = PROJECT_ROOT / 'models' / 'deploy_bundle.joblib'
META_PATH = PROJECT_ROOT / 'models' / 'deploy_meta.json'
DEFAULTS_PATH = PROJECT_ROOT / 'models' / 'default_employee.json'


def main():
    df = prepare_training_frame(with_engineering=True)
    state, x_train, y_train = fit_preprocessor(df, with_engineering=True)

    model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **FINAL_RF_PARAMS)
    model.fit(x_train, y_train)

    train_proba = model.predict_proba(x_train)[:, 1]
    train_metrics = classification_metrics(y_train, model.predict(x_train), train_proba)

    bundle = {
        'model': model,
        'preprocessor': state,
        'feature_columns': state.num_cols + state.ord_cols + state.nom_cols,
    }
    BUNDLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, BUNDLE_PATH)

    meta = {
        'model_name': 'RandomForest_tuned',
        'params': FINAL_RF_PARAMS,
        'n_features': len(state.feature_names),
        'train_metrics': train_metrics,
        'risk_thresholds': {'low': 0.35, 'high': 0.55},
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

    defaults = default_employee_row(df)
    DEFAULTS_PATH.write_text(json.dumps(defaults, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Saved: {BUNDLE_PATH}')
    print(f'Saved: {META_PATH}')
    print(f'Saved: {DEFAULTS_PATH}')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
