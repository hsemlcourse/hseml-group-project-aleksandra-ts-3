from pathlib import Path

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
DATA_RAW = DATA_RAW_DIR / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DEFAULT_CSV_NAME = 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
TRAIN_FRACTION = 0.64
VAL_FRACTION = 0.16
TARGET_COL = 'Attrition'
DROP_ALWAYS = {
    'EmployeeCount',
    'Over18',
    'StandardHours',
    'EmployeeNumber',
}
