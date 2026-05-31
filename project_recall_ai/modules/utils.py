import os
from pathlib import Path

def ensure_data_dirs(base='data'):
    Path(base, 'uploads').mkdir(parents=True, exist_ok=True)
    Path(base, 'memories').mkdir(parents=True, exist_ok=True)

def safe_filename(name: str) -> str:
    return ''.join(c for c in name if c.isalnum() or c in ('_', '-')).rstrip()
