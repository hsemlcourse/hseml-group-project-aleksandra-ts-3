# ML Project — Прогноз оттока сотрудников (IBM HR Attrition)

**Студент:** Целовальникова Александра Тимофеевна

**Группа:** БИВ231

## Описание задачи

**Задача:** бинарная классификация: предсказать вероятность увольнения (`Attrition`), ранжировать сотрудников по риску

**Датасет:** [IBM HR Analytics Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset): 1470 строк, 35 признаков (после загрузки в `data/raw/`). Найден через Kaggle по запросам связанными в HR, выбран как стандартный размеченный табличный кейс

**Метрики:** основной упор на **ROC-AUC** и **Recall** (так как дисбаланс), ну и дополнительно **Precision**, **F1**, **Accuracy**

## Структура репозитория

```
.
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   ├── modeling.py             # Обучение и оценка моделей
│   ├── config.py               # Настройки проекта
│   ├── hypothesis_tests.py     # Статистические проверки для EDA
│   └── scripts/
│       ├── download_data.py    # Загрузка датасета в data/raw
│       ├── register_kernel.py  # Регистрация jupyter-kernel
│       └── make_synthetic_train.py # Синтетическое расширение train
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
├── pyproject.toml
├── Dockerfile                  # Контейнер
├── docker-compose.yml
└── README.md
```

## Запуск

```bash
# 1. Клонировать репозиторий
git clone https://github.com/hsemlcourse/hseml-group-project-aleksandra-ts
cd hseml-group-project-aleksandra-ts
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

python -m pip install -r requirements.txt
python -m src.scripts.download_data
python -m pytest tests/ -q
```

### Docker

```bash
docker compose build
docker compose run --rm app
```

Образ выполняет `download_data` и `pytest` (см. `Dockerfile`)

### Запустить ноутбуки

Пошагово:

1. Проверить интерпретатор: `.venv\Scripts\python.exe` (Windows) или `.venv/bin/python` (Linux/macOS)
2. Если kernel не виден, зарегистрировать:
   ```bash
   python -m src.scripts.register_kernel
   ```
3. Открыть ноутбук и выберать kernel `Python (hseml-attrition)`
4. Запуск ноутбуков по порядку:
   - `notebooks/01_eda.ipynb`
   - `notebooks/02_baseline.ipynb`
   - `notebooks/03_experiments.ipynb`
5. После `03_experiments.ipynb` проверить, что артефакты модели появились в `models/`

## Данные

- Исходный CSV: `WA_Fn-UseC_-HR-Employee-Attrition.csv` в `data/raw/` **или** выполните `python -m src.scripts.download_data`.
- Сплит **64% / 16% / 20%** (train / val / test), стратификация по `Attrition`. Утечки по времени нет: срез анкеты, нет временных рядов: используется случайное стратифицированное разбиение

### Синтетическое расширение train (10 000+)

```bash
python -m src.scripts.make_synthetic_train
```

- делает стратифицированный split train/val/test
- расширяет train синтетическими строками до 10 000 (bootstrap + малый шум по числовым признакам)
- не изменяет val/test
- сохраняет:
  - `data/processed/train_synthetic_10000.csv`
  - `data/processed/synthetic_metrics.json` (метрики до/после)

## Результаты (CP1)

В таблице метрики после прогона `02_baseline.ipynb` и `03_experiments.ipynb`.
Приоритет выбора: `ROC-AUC` (ранжирование риска) и `Recall` по классу `Attrition=1`

**Baseline** (без FE):

| Модель                   | ROC-AUC (val/test) | Recall (val/test) | F1 (val/test) |
| ------------------------ | ------------------ | ----------------- | ------------- |
| Baseline LogReg (без FE) | 0.871 / 0.774      | 0.816 / 0.617     | 0.534 / 0.460 |

**Сводка:**

| Модель                    | ROC-AUC val | ROC-AUC test | Примечание |
| ------------------------- | ----------: | -----------: | ---------- |
| RandomForest_tuned        | **0.874**   | 0.721        | `RandomizedSearchCV`, `n_estimators=700`, `max_depth=6`, `min_samples_leaf=8`, `min_samples_split=10` |
| LogReg + FE               | 0.872       | —            |            |
| RandomForest + FE         | 0.865       | —            |            |
| PCA95 + LogReg            | 0.853       | 0.775        |            |
| Voting_soft               | 0.851       | —            |            |
| XGBoost + FE              | 0.834       | —            |            |
| LightGBM + FE             | 0.819       | —            |            |
| KNN + FE                  | 0.783       | —            | k=15, веса по расстоянию |

**Финальный выбор для CP1:** `RandomForest tuned`: лучший `val ROC-AUC` среди проверенных конфигураций при заметно лучшем `val Recall`, чем у дефолтного RandomForest

Артефакты финальной модели сохранены в `models/`

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
