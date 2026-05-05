import pandas as pd
from pathlib import Path

def load_csv(file_path: str) -> pd.DataFrame:
    validate_file(file_path) 
    df = pd.read_csv(file_path)
    return df

def validate_file(file_path: str) -> str: 
    path = Path(file_path)
    if not path.exists(): 
        raise FileNotFoundError("Path not found")
    if path.suffix.lower() != ".csv": 
        raise ValueError("file not supported")
    if path.stat().st_size == 0: 
        raise ValueError("empty file")

