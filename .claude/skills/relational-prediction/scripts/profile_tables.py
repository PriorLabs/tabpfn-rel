"""Profile every column of every CSV or Parquet table in a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MISSING_MARKERS = {"", "\\N", "NULL", "null", "None", "none", "NaN", "nan", "N/A"}
MISSING_MARKERS |= {"n/a", "NA", "-", "?", "unknown", "Unknown"}


def read_raw(path: Path) -> pd.DataFrame:
    """Read a table as written, keeping missing-value markers as strings."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])


def profile_column(col: pd.Series) -> dict[str, object]:
    """Summarize one column over all its rows."""
    as_text = col.dropna().astype(str)
    markers = as_text[as_text.str.strip().isin(MISSING_MARKERS)].value_counts()
    top = as_text.value_counts().head(3)
    row: dict[str, object] = {
        "column": col.name,
        "dtype": str(col.dtype),
        "null_share": round(col.isna().mean(), 4),
        "n_unique": as_text.nunique(),
        "top_values": "; ".join(f"{v[:30]!r} x{n}" for v, n in top.items()),
        "markers": "; ".join(f"{v!r} x{n}" for v, n in markers.items()),
    }
    values = as_text[~as_text.str.strip().isin(MISSING_MARKERS)]
    numbers = pd.to_numeric(values.str.strip(), errors="coerce")
    if len(values) and numbers.notna().mean() > 0.95:
        row["min"], row["max"] = numbers.min(), numbers.max()
        row["zeros"], row["negatives"] = (
            int((numbers == 0).sum()),
            int((numbers < 0).sum()),
        )
    elif len(values) and values.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.95:
        dates = pd.to_datetime(values, errors="coerce", format="mixed")
        row["min"], row["max"] = dates.min(), dates.max()
    return row


def main() -> None:
    """Print a per-column profile for each table under the given directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    paths = sorted(
        p for p in args.directory.rglob("*") if p.suffix in {".csv", ".parquet"}
    )
    if not paths:
        raise SystemExit(f"No .csv or .parquet files under {args.directory}.")
    with pd.option_context("display.width", 250, "display.max_colwidth", 80):
        for path in paths:
            df = read_raw(path)
            print(f"\n## {path.relative_to(args.directory)}: {len(df):,} rows")
            profile = pd.DataFrame([profile_column(df[c]) for c in df.columns])
            print(profile.fillna("").to_string(index=False))


if __name__ == "__main__":
    main()
