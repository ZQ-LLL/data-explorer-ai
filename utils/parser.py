"""
utils/parser.py
Responsible for: file parsing, DataFrame metadata extraction, large dataset sampling.
No Streamlit imports — pure data logic, fully testable in isolation.
"""

import io

import pandas as pd

# Rows above this threshold trigger automatic sampling
SAMPLE_THRESHOLD = 50_000


def parse_file(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    """
    Parse an uploaded file based on its extension.
    Returns (DataFrame, error_message). On success, error_message is empty.
    Supports CSV (utf-8 and latin-1), JSON (array and JSON Lines), Excel.
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin-1")

        elif name.endswith(".json"):
            raw = uploaded_file.read()
            try:
                df = pd.read_json(io.BytesIO(raw), orient="records")
            except ValueError:
                df = pd.read_json(io.BytesIO(raw), lines=True)

        elif name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(uploaded_file, engine="openpyxl")

        else:
            ext = uploaded_file.name.split(".")[-1]
            return None, f"Unsupported file format: .{ext}"

        return df, ""

    except Exception as e:
        return None, f"Failed to parse file: {e}"


def is_meaningful_numeric(series: pd.Series) -> bool:
    """
    Return True only if a numeric column is worth running descriptive stats on.
    Excludes:
      - ID-like columns (name is 'id' or ends with 'id')
      - Boolean-encoded columns (only 0/1 values)
    """
    name_lower = series.name.lower()
    if name_lower == "id" or name_lower.endswith("id"):
        return False
    unique_vals = set(series.dropna().unique())
    if unique_vals <= {0, 1}:
        return False
    return True


def describe_dataframe(df: pd.DataFrame) -> dict:
    """
    Extract key metadata from a DataFrame for use in AI prompts and UI display.
    Returns a plain dict — no Streamlit dependencies.
    """
    raw_num_cols = df.select_dtypes(include="number").columns.tolist()
    num_cols = [c for c in raw_num_cols if is_meaningful_numeric(df[c])]
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    missing = int(df.isnull().sum().sum())
    missing_pct = round(missing / (df.shape[0] * df.shape[1]) * 100, 1) if df.size else 0

    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "numeric_cols": num_cols,
        "text_cols": cat_cols,
        "missing_total": missing,
        "missing_pct": f"{missing_pct}%",
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def maybe_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """
    Return a sampled DataFrame if rows exceed SAMPLE_THRESHOLD, plus a bool flag.
    The flag lets the UI show a warning when sampling is active.
    """
    if len(df) > SAMPLE_THRESHOLD:
        return df.sample(n=SAMPLE_THRESHOLD, random_state=42), True
    return df, False


def df_to_bytes(df: pd.DataFrame, fmt: str) -> tuple[bytes, str, str]:
    """
    Convert a DataFrame to downloadable bytes.
    Returns (bytes, mime_type, file_extension).
    """
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8"), "text/csv", "csv"
    elif fmt == "json":
        return df.to_json(orient="records", indent=2).encode("utf-8"), "application/json", "json"
    else:  # excel
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        return (
            buf.getvalue(),
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "xlsx",
        )
