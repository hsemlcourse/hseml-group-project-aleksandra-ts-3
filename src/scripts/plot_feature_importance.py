import joblib
import matplotlib.pyplot as plt

from src.config import PROJECT_ROOT

BUNDLE_PATH = PROJECT_ROOT / 'models' / 'deploy_bundle.joblib'
OUT_PATH = PROJECT_ROOT / 'report' / 'images' / 'feature_importance.png'
TOP_N = 15


def main():
    if not BUNDLE_PATH.is_file():
        raise FileNotFoundError(f'Нет bundle: {BUNDLE_PATH}')

    payload = joblib.load(BUNDLE_PATH)
    model = payload['model']
    names = payload['preprocessor'].feature_names
    importances = model.feature_importances_

    order = importances.argsort()[::-1][:TOP_N]
    top_names = [names[i] for i in order]
    top_vals = importances[order]

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = ['#517D8B' if i < top_vals.max() * 0.7 else '#1B262C' for i in top_vals]
    ax.barh(top_names[::-1], top_vals[::-1], color=colors[::-1])
    ax.set_xlabel('Важность (Gini)')
    ax.set_title('Топ-15 признаков: RandomForest tuned')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150)
    plt.close(fig)
    print(f'Saved: {OUT_PATH}')


if __name__ == '__main__':
    main()
