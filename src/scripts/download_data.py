import sys
import urllib.request
from pathlib import Path

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
    print(f'Downloading to {dest} ...')
    urllib.request.urlretrieve(MIRROR_URL, dest)
    print('Done')
    return 0

if __name__ == '__main__':
    sys.exit(main())
