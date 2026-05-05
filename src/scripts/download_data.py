import sys
import urllib.request
from pathlib import Path

from src.attrition_parser import parse_ibm_attrition_csv

MIRROR_URL = (
    'https://raw.githubusercontent.com/nelson-wu/employee-attrition-ml/master/'
    'WA_Fn-UseC_-HR-Employee-Attrition.csv'
)


def main():
    root = Path(__file__).resolve().parents[2]
    raw_dir = root / 'data' / 'raw'
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
    if dest.exists():
        print(f'Already present: {dest}')
        return 0
    print(f'Downloading from {MIRROR_URL} ...')
    with urllib.request.urlopen(MIRROR_URL, timeout=120) as response:
        body = response.read().decode('utf-8', errors='replace')
    df = parse_ibm_attrition_csv(body)
    print(f'Parsed CSV rows={len(df)}, cols={len(df.columns)}')
    df.to_csv(dest, index=False)
    print(f'Saved: {dest}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
