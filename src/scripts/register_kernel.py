import subprocess
import sys


def main():
    display_name = 'Python (hseml-attrition)'
    kernel_name = 'hseml-attrition'
    cmd = [
        sys.executable,
        '-m',
        'ipykernel',
        'install',
        '--user',
        f'--name={kernel_name}',
        f'--display-name={display_name}',
    ]
    print(' ', ' '.join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print('Done. Select this kernel in notebook:', display_name)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
