from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pandas as pd

TARGET_ALIASES = {
    "churn",
    "churn_flag",
    "is_churn",
    "churned",
    "target",
}


def _find_csv_in_zip(path: Path) -> str:
    with ZipFile(path) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".csv")
        ]

        if not names:
            raise ValueError("No CSV found in ZIP")

        return names[0]


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".zip":
        name = _find_csv_in_zip(path)

        with ZipFile(path) as archive, archive.open(name) as handle:
            return pd.read_csv(handle)

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported dataset format: {suffix}"
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in out.columns
    ]
    return out


def find_target_column(
    df: pd.DataFrame,
    requested: str | None = None,
) -> str:
    columns = set(df.columns)

    if requested:
        requested = (
            requested.strip()
            .lower()
            .replace(" ", "_")
        )

        if requested not in columns:
            raise ValueError(
                f"Target column '{requested}' not found"
            )

        return requested

    for column in TARGET_ALIASES:
        if column in columns:
            return column

    raise ValueError(
        "Could not infer churn target column"
    )


def encode_target(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    if numeric.notna().all():
        values = set(
            numeric.astype(int).unique()
        )
        if values.issubset({0, 1}):
            return numeric.astype(int)

    mapping = {
        "yes": 1,
        "y": 1,
        "true": 1,
        "churn": 1,
        "churned": 1,
        "1": 1,
        "no": 0,
        "n": 0,
        "false": 0,
        "active": 0,
        "retained": 0,
        "0": 0,
    }

    encoded = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    if encoded.isna().any():
        values = (
            series[encoded.isna()]
            .drop_duplicates()
            .tolist()[:10]
        )
        raise ValueError(
            f"Unsupported target values: {values}"
        )

    return encoded.astype(int)


def prepare_xy(
    df: pd.DataFrame,
    target: str | None = None,
):
    frame = normalize_columns(df)
    frame = frame.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    target_column = find_target_column(
        frame,
        target,
    )
    y = encode_target(
        frame.pop(target_column)
    )

    excluded = {
        "id",
        "customer_id",
        "mobile_number",
        "msisdn",
    }

    frame = frame.drop(
        columns=[
            column
            for column in excluded
            if column in frame.columns
        ],
        errors="ignore",
    )

    return frame, y, target_column
