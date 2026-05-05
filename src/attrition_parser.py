import csv
import io

import pandas as pd


def parse_ibm_attrition_csv(text):
    stream = io.StringIO(text.strip())
    reader = csv.DictReader(stream)
    rows = list(reader)
    if not rows:
        raise ValueError('CSV: пустой файл или нет заголовка')
    return pd.DataFrame(rows)
