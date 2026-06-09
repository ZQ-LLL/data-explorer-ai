"""
Data Explorer — M1: File Upload + Multi-format Parsing + Data Preview
"""

import streamlit as st
import pandas as pd
import io

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Data Explorer",
    page_icon="🔍",
    layout="wide",
)

# ── Style ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .stat-label {
        font-size: 0.78rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #111827;
    }
    .info-box {
        background: #eff6ff;
        border-left: 3px solid #3b82f6;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        color: #1e40af;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Parsing ───────────────────────────────────────────────────
def parse_file(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    """
    Parse an uploaded file based on its extension.
    Returns (DataFrame, error_message). On success, error_message is empty.
    """
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            # Try UTF-8 first, fall back to latin-1 for non-ASCII files
            try:
                df = pd.read_csv(uploaded_file, encoding="utf-8")
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding="latin-1")

        elif name.endswith(".json"):
            raw = uploaded_file.read()
            # Support both JSON array and JSON Lines formats
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
      - ID-like columns (name contains 'id' as a whole word)
      - Boolean-like columns (only 0/1/NaN values)
    """
    name_lower = series.name.lower()
    # Reject if column name is or ends with 'id' (e.g. 'id', 'jobId', 'company_id')
    if name_lower == "id" or name_lower.endswith("id"):
        return False
    # Reject if the only non-null values are 0 and 1 (boolean encoded as int)
    unique_vals = set(series.dropna().unique())
    if unique_vals <= {0, 1}:
        return False
    return True


def describe_dataframe(df: pd.DataFrame) -> dict:
    """
    Extract key metadata from a DataFrame.
    This dict is stored in session_state and reused by the M2 AI report.
    """
    raw_num_cols = df.select_dtypes(include="number").columns.tolist()
    # Filter out ID columns and boolean-encoded columns
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


# ── Main UI ───────────────────────────────────────────────────
st.markdown('<p class="hero-title">🔍 Data Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload a dataset and let AI help you understand it.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Supports CSV · JSON · Excel",
    type=["csv", "json", "xlsx", "xls"],
    help="Recommended file size: under 200 MB",
)

# ── File loaded: parse + display ─────────────────────────────
if uploaded_file is not None:
    df, error = parse_file(uploaded_file)

    if error:
        st.error(error)
        st.stop()

    # Store in session_state so M2/M3 can access without re-parsing
    st.session_state["df"] = df
    st.session_state["filename"] = uploaded_file.name

    meta = describe_dataframe(df)
    st.session_state["meta"] = meta  # M2 will use this to build the AI prompt

    st.success(f"✅ Loaded: **{uploaded_file.name}**")

    # ── Summary cards ─────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "Rows",         f"{meta['rows']:,}"),
        (c2, "Columns",      f"{meta['columns']}"),
        (c3, "Numeric cols", f"{len(meta['numeric_cols'])}"),
        (c4, "Missing",      meta["missing_pct"]),
    ]
    for col, label, value in cards:
        col.markdown(
            f'<div class="stat-card"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # ── Column overview ───────────────────────────────────────
    with st.expander("📋 Column Overview", expanded=True):
        col_info = pd.DataFrame({
            "Column":    df.columns,
            "Type":      df.dtypes.astype(str).values,
            "Non-null":  df.count().values,
            "Missing":   df.isnull().sum().values,
            "Sample":    [
                str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0] > 0 else "(all null)"
                for c in df.columns
            ],
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)

    # ── Raw data preview ──────────────────────────────────────
    with st.expander("🗂 Raw Data Preview (first 100 rows)", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    # ── Descriptive statistics with interactive column selector ──
    all_num_cols = df.select_dtypes(include="number").columns.tolist()
    if all_num_cols:
        with st.expander("📊 Descriptive Statistics", expanded=False):
            st.caption("Select which numeric columns to include. Columns likely to be IDs or flags are unchecked by default.")
            selected_cols = []
            # Render one checkbox per numeric column; default = passed the meaningful filter
            cols_per_row = 4
            rows = [all_num_cols[i:i+cols_per_row] for i in range(0, len(all_num_cols), cols_per_row)]
            for row in rows:
                cb_cols = st.columns(cols_per_row)
                for cb_col, col_name in zip(cb_cols, row):
                    default = is_meaningful_numeric(df[col_name])
                    if cb_col.checkbox(col_name, value=default, key=f"stat_cb_{col_name}"):
                        selected_cols.append(col_name)

            if selected_cols:
                st.dataframe(
                    df[selected_cols].describe().round(2),
                    use_container_width=True,
                )
            else:
                st.info("Check at least one column above to see statistics.")

    # ── Sample record ─────────────────────────────────────────
    with st.expander("🪪 Sample Record (first non-null row)", expanded=True):
        # Pick the first row that has the most non-null values
        best_row_idx = df.isnull().sum(axis=1).idxmin()
        sample = df.loc[best_row_idx]

        # Display as a two-column key/value grid
        def is_nonempty(v) -> bool:
            """Safely check if a cell value is non-null and non-empty."""
            try:
                return pd.notna(v) and str(v).strip() != ""
            except (ValueError, TypeError):
                # pd.notna() raises ValueError on array-like values; treat those as non-empty
                return True

        fields = [(k, v) for k, v in sample.items() if is_nonempty(v)]
        empty_fields = [(k, v) for k, v in sample.items() if not is_nonempty(v)]

        if fields:
            # Render non-null fields in a styled grid
            grid_html = '<div style="display:grid;grid-template-columns:220px 1fr;gap:0.3rem 1rem;">'
            for k, v in fields:
                val_str = str(v)
                # Truncate very long values (e.g. descriptionText)
                display_val = val_str[:200] + "…" if len(val_str) > 200 else val_str
                grid_html += (
                    f'<div style="font-size:0.78rem;color:#6b7280;font-weight:600;'
                    f'padding:0.35rem 0;border-bottom:1px solid #f3f4f6;word-break:break-word;">{k}</div>'
                    f'<div style="font-size:0.88rem;color:#111827;padding:0.35rem 0;'
                    f'border-bottom:1px solid #f3f4f6;word-break:break-word;">{display_val}</div>'
                )
            grid_html += "</div>"
            st.markdown(grid_html, unsafe_allow_html=True)

        if empty_fields:
            st.caption(f"⚠ {len(empty_fields)} field(s) are empty in this record: "
                       + ", ".join(k for k, _ in empty_fields[:10])
                       + ("…" if len(empty_fields) > 10 else ""))

    # ── M2 handoff notice ─────────────────────────────────────
    st.markdown("""
    <div class="info-box">
    ✨ <b>M1 complete</b> — Data loaded successfully.
    Next up (M2): connect the Claude API to auto-generate a dataset analysis report.
    </div>
    """, unsafe_allow_html=True)

else:
    # Empty state
    st.markdown("""
    <div style="text-align:center; padding: 3rem 0; color: #9ca3af;">
        <div style="font-size: 3rem;">📂</div>
        <div style="font-size: 1rem; margin-top: 0.5rem;">Upload a file above to get started</div>
    </div>
    """, unsafe_allow_html=True)
