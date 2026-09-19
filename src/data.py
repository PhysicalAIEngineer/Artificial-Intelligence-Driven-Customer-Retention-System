from __future__ import annotations
from pathlib import Path
from zipfile import ZipFile
import pandas as pd

TARGET_ALIASES={"churn","churn_flag","is_churn","churned","target"}

def _find_csv_in_zip(path):
    with ZipFile(path) as z:
        names=[n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names: raise ValueError("No CSV found in ZIP")
        return names[0]

def load_dataframe(path):
    path=Path(path)
    if path.suffix.lower()==".zip":
        name=_find_csv_in_zip(path)
        with ZipFile(path) as z, z.open(name) as f: return pd.read_csv(f)
    if path.suffix.lower()==".csv": return pd.read_csv(path)
    if path.suffix.lower() in {".parquet",".pq"}: return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")

def normalize_columns(df):
    out=df.copy()
    out.columns=[str(c).strip().lower().replace(" ","_") for c in out.columns]
    return out

def find_target_column(df,requested=None):
    cols=set(df.columns)
    if requested:
        requested=requested.strip().lower().replace(" ","_")
        if requested not in cols: raise ValueError(f"Target column '{requested}' not found")
        return requested
    for c in TARGET_ALIASES:
        if c in cols: return c
    raise ValueError("Could not infer churn target column")

def encode_target(s):
    numeric=pd.to_numeric(s,errors="coerce")
    if numeric.notna().all() and set(numeric.astype(int).unique()).issubset({0,1}): return numeric.astype(int)
    mapping={"yes":1,"y":1,"true":1,"churn":1,"churned":1,"1":1,"no":0,"n":0,"false":0,"active":0,"retained":0,"0":0}
    out=s.astype(str).str.strip().str.lower().map(mapping)
    if out.isna().any(): raise ValueError(f"Unsupported target values: {s[out.isna()].drop_duplicates().tolist()[:10]}")
    return out.astype(int)

def prepare_xy(df,target=None):
    frame=normalize_columns(df).replace([float("inf"),float("-inf")],pd.NA)
    target_column=find_target_column(frame,target); y=encode_target(frame.pop(target_column))
    frame=frame.drop(columns=[c for c in ["id","customer_id","mobile_number","msisdn"] if c in frame.columns],errors="ignore")
    return frame,y,target_column
