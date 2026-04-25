from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import RANDOM_STATE

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None


def classification_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    if y_proba is not None and y_proba.shape[0] == y_true.shape[0]:
        try:
            metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics['roc_auc'] = float('nan')
    return metrics


def make_baseline_logistic():
    return LogisticRegression(
        class_weight='balanced',
        max_iter=2000,
        random_state=RANDOM_STATE,
        solver='lbfgs',
    )


def make_knn():
    return KNeighborsClassifier(n_neighbors=15, weights='distance')


def make_gaussian_nb():
    return GaussianNB()


def make_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_gradient_boosting():
    return GradientBoostingClassifier(random_state=RANDOM_STATE)


def make_xgboost():
    if xgb is None:
        raise ImportError('xgboost is not installed')
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def make_lightgbm():
    if lgb is None:
        raise ImportError('lightgbm is not installed')
    return lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def make_voting_ensemble():
    return VotingClassifier(
        estimators=[
            ('rf', make_random_forest()),
            ('lgb', make_lightgbm()),
            ('gbc', make_gradient_boosting()),
        ],
        voting='soft',
        n_jobs=-1,
    )


def pca_logistic_pipeline(n_components=0.95):
    return Pipeline(
        [
            ('scaler2', StandardScaler()),
            ('pca', PCA(n_components=n_components, random_state=RANDOM_STATE)),
            (
                'clf',
                LogisticRegression(
                    class_weight='balanced',
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def pca_rf_pipeline(n_components=0.95):
    return Pipeline(
        [
            ('scaler2', StandardScaler()),
            ('pca', PCA(n_components=n_components, random_state=RANDOM_STATE)),
            (
                'clf',
                RandomForestClassifier(
                    n_estimators=300,
                    class_weight='balanced_subsample',
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def stratified_kfold():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def run_randomized_search(model, param_distributions, X, y, *, n_iter=25, scoring='roc_auc'):
    return RandomizedSearchCV(
        model,
        param_distributions,
        n_iter=n_iter,
        cv=stratified_kfold(),
        scoring=scoring,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True,
    ).fit(X, y)
