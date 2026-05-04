[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
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

**Сводка:**

| Модель                                  | ROC-AUC (val/test) | Recall val/test | F1 val/test       | Примечание                                                                |
| --------------------------------------- | ------------------ | --------------- | ----------------- | ------------------------------------------------------------------------- |
| Baseline LogReg (без FE)                | 0.871 / 0.774      | 0.816 / 0.617   | 0.534 / 0.460     | `02_baseline.ipynb`                                                       |
| LogReg + FE                             | 0.872 / 0.767      | 0.789 / 0.617   | 0.545 / 0.439     | `03_experiments.ipynb`                                                    |
| RandomForest + FE                       | 0.865 / 0.718      | 0.158 / 0.128   | 0.267 / 0.211     | `03_experiments.ipynb`                                                    |
| XGBoost + FE                            | 0.830 / 0.730      | 0.342 / 0.191   | 0.491 / 0.269     | `03_experiments.ipynb`                                                    |
| LightGBM + FE                           | 0.819 / 0.714      | 0.421 / 0.255   | 0.525 / 0.312     | `03_experiments.ipynb`                                                    |
| RandomForest tuned (RandomizedSearchCV) | **0.874 / 0.720**  | 0.579 / 0.362   | **0.557 / 0.366** | `n_estimators=700, max_depth=6, min_samples_leaf=8, min_samples_split=10` |

**Финальный выбор:** `RandomForest tuned`: лучший `val ROC-AUC` среди проверенных конфигураций при заметно лучшем `val Recall`, чем у дефолтного RandomForest

Артефакты финальной модели сохранены в `models/`

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
