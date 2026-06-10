"""
Data Explorer — M5: Layout redesign, sidebar controls, data editing, large dataset optimization
"""

import io
import json
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

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
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.1rem;
    }
    .hero-sub {
        color: #6b7280;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    .stat-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        text-align: center;
    }
    .stat-label {
        font-size: 0.7rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #111827;
    }
    .report-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        font-size: 0.92rem;
        line-height: 1.75;
        color: #1e293b;
    }
    .chart-label {
        font-size: 0.78rem;
        color: #6b7280;
        font-style: italic;
        margin-top: 0.2rem;
    }
    .board-section {
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        background: #ffffff;
    }
    .board-section-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.6rem;
    }
    .pinned-label {
        font-size: 0.75rem;
        color: #9ca3af;
        margin-bottom: 0.3rem;
    }
    /* Make right chat column feel distinct */
    .chat-col-header {
        font-size: 1rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


# ── AI client (supports both .env key and user-provided key) ──
def get_ai_client(user_key: str = "", model_base: str = "openrouter") -> OpenAI | None:
    """
    Build an OpenAI-compatible client.
    User-provided key takes priority over .env key.
    """
    api_key = user_key.strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


# ── Parsing ───────────────────────────────────────────────────
def parse_file(uploaded_file) -> tuple[pd.DataFrame | None, str]:
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
    name_lower = series.name.lower()
    if name_lower == "id" or name_lower.endswith("id"):
        return False
    unique_vals = set(series.dropna().unique())
    if unique_vals <= {0, 1}:
        return False
    return True


def describe_dataframe(df: pd.DataFrame) -> dict:
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


# ── Large dataset sampling ────────────────────────────────────
SAMPLE_THRESHOLD = 50_000

def maybe_sample(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Return a sampled DataFrame if rows exceed threshold, plus a flag."""
    if len(df) > SAMPLE_THRESHOLD:
        return df.sample(n=SAMPLE_THRESHOLD, random_state=42), True
    return df, False


# ── AI prompt builders ────────────────────────────────────────
def build_analysis_prompt(filename: str, meta: dict, df: pd.DataFrame) -> str:
    stats_lines = []
    if meta["numeric_cols"]:
        stats = df[meta["numeric_cols"]].describe().round(2)
        for col in meta["numeric_cols"]:
            s = stats[col]
            stats_lines.append(
                f"  - {col}: mean={s['mean']}, min={s['min']}, max={s['max']}, "
                f"missing={df[col].isnull().sum()}"
            )
    missing_by_col = df.isnull().sum().sort_values(ascending=False)
    top_missing = missing_by_col[missing_by_col > 0].head(8)
    missing_lines = [f"  - {col}: {cnt} missing" for col, cnt in top_missing.items()]

    return f"""You are a data analyst. A user has uploaded a dataset and needs a clear, insightful report.

Dataset: {filename}
Shape: {meta['rows']} rows × {meta['columns']} columns
Overall missing rate: {meta['missing_pct']}

All columns ({meta['columns']} total):
{', '.join(meta['column_names'])}

Meaningful numeric columns:
{chr(10).join(stats_lines) if stats_lines else '  (none detected)'}

Columns with most missing values:
{chr(10).join(missing_lines) if missing_lines else '  (no missing values)'}

Text/categorical columns:
{', '.join(meta['text_cols']) if meta['text_cols'] else '(none)'}

Please write a concise dataset analysis report (around 200-300 words) covering:
1. What this dataset likely contains and its probable purpose
2. Key observations about data structure and quality
3. Notable patterns or potential issues worth investigating
4. 2-3 specific analysis questions this dataset could answer

Write in clear, plain English. Be specific — reference actual column names and numbers. Always respond in English.
"""


def build_chart_prompt(meta: dict, df: pd.DataFrame) -> str:
    cat_samples = {}
    for col in meta["text_cols"][:8]:
        top_vals = df[col].dropna().value_counts().head(5).index.tolist()
        cat_samples[col] = top_vals

    return f"""You are a data visualization expert. Recommend exactly 3 or 4 charts for this dataset.

Dataset info:
- Shape: {meta['rows']} rows × {meta['columns']} columns
- Numeric columns: {', '.join(meta['numeric_cols']) or 'none'}
- Text/categorical columns: {', '.join(meta['text_cols'][:15]) or 'none'}
- Top category values: {json.dumps(cat_samples, ensure_ascii=False)}

Supported chart types: bar, histogram, scatter, box, pie

Rules:
- Only use columns that actually exist in the dataset
- For bar: x=categorical, y=numeric or "count"
- For histogram: x=numeric
- For scatter: x and y both numeric
- For box: x=categorical, y=numeric
- For pie: names=low-cardinality categorical (under 10 unique values)
- Prefer columns with fewer missing values

Respond with ONLY a valid JSON array, no explanation, no markdown fences:
[
  {{
    "type": "bar",
    "title": "Chart title",
    "x": "column_name",
    "y": "column_name_or_count",
    "color": null,
    "reason": "Why this chart is useful",
    "top_n": 10
  }}
]"""


# ── AI report generator (streaming) ──────────────────────────
def generate_report(client: OpenAI, prompt: str) -> str:
    stream = client.chat.completions.create(
        model=st.session_state.get("selected_model", "anthropic/claude-haiku-4-5"),
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=1024,
    )
    placeholder = st.empty()
    full_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_text += delta
        placeholder.markdown(f'<div class="report-box">{full_text}▌</div>', unsafe_allow_html=True)
    placeholder.markdown(f'<div class="report-box">{full_text}</div>', unsafe_allow_html=True)
    return full_text


# ── Chart spec renderer ───────────────────────────────────────
def render_chart(spec: dict, df: pd.DataFrame, chart_key: str = "") -> go.Figure | None:
    """Render a chart from a spec dict. Returns the Figure or None on failure."""
    chart_type = spec.get("type", "").lower()
    title = spec.get("title", "Chart")
    x_col = spec.get("x")
    y_col = spec.get("y")
    color_col = spec.get("color")
    top_n = spec.get("top_n")
    reason = spec.get("reason", "")

    required_cols = [c for c in [x_col, y_col, color_col] if c and c != "count"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        st.warning(f"Skipped '{title}': column(s) {missing_cols} not found.")
        return None

    plot_df = df[required_cols].dropna().copy() if required_cols else df.copy()

    try:
        if chart_type == "bar":
            if y_col == "count" or y_col is None:
                counts = plot_df[x_col].value_counts().reset_index()
                counts.columns = [x_col, "count"]
                if top_n:
                    counts = counts.head(top_n)
                fig = px.bar(counts, x=x_col, y="count", title=title)
            else:
                agg = plot_df.groupby(x_col)[y_col].mean().reset_index()
                if top_n:
                    agg = agg.nlargest(top_n, y_col)
                fig = px.bar(agg, x=x_col, y=y_col, title=title)
        elif chart_type == "histogram":
            fig = px.histogram(plot_df, x=x_col, title=title)
        elif chart_type == "scatter":
            fig = px.scatter(plot_df, x=x_col, y=y_col, title=title,
                             color=color_col if color_col else None)
        elif chart_type == "box":
            fig = px.box(plot_df, x=x_col, y=y_col, title=title)
        elif chart_type == "pie":
            counts = plot_df[x_col].value_counts().reset_index()
            counts.columns = [x_col, "count"]
            if top_n:
                counts = counts.head(top_n)
            fig = px.pie(counts, names=x_col, values="count", title=title)
        else:
            st.warning(f"Unknown chart type: {chart_type}")
            return None

        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            font_family="Inter, sans-serif",
            title_font_size=14,
            margin=dict(t=45, b=35, l=35, r=15),
        )
        unique_key = chart_key or f"chart_{hash(title)}_{chart_type}"
        st.plotly_chart(fig, use_container_width=True, key=unique_key)
        if reason:
            st.markdown(f'<div class="chart-label">💡 {reason}</div>', unsafe_allow_html=True)
        return fig

    except Exception as e:
        st.warning(f"Could not render '{title}': {e}")
        return None


def get_chart_specs(client: OpenAI, meta: dict, df: pd.DataFrame, cache_key: str) -> list[dict] | None:
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    try:
        response = client.chat.completions.create(
            model=st.session_state.get("selected_model", "anthropic/claude-haiku-4-5"),
            messages=[{"role": "user", "content": build_chart_prompt(meta, df)}],
            stream=False,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        specs = json.loads(raw)
        st.session_state[cache_key] = specs
        return specs
    except Exception as e:
        st.error(f"Failed to get chart recommendations: {e}")
        return None


# ── Data export helper ────────────────────────────────────────
def df_to_bytes(df: pd.DataFrame, fmt: str) -> tuple[bytes, str, str]:
    """Convert DataFrame to downloadable bytes. Returns (bytes, mime, extension)."""
    if fmt == "csv":
        return df.to_csv(index=False).encode("utf-8"), "text/csv", "csv"
    elif fmt == "json":
        return df.to_json(orient="records", indent=2).encode("utf-8"), "application/json", "json"
    else:  # excel
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        return buf.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Data Explorer")
    st.markdown("---")

    # File upload
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        label="CSV · JSON · Excel",
        type=["csv", "json", "xlsx", "xls"],
        help="Recommended: under 200 MB",
        label_visibility="collapsed",
    )

    st.markdown("---")

    # API settings
    st.markdown("### ⚙️ AI Settings")
    user_api_key = st.text_input(
        "API Key (OpenRouter)",
        type="password",
        placeholder="sk-or-v1-… (or set in .env)",
        help="Your key takes priority over the .env file.",
    )

    MODEL_OPTIONS = {
        "Claude Haiku 4.5 (fast)":   "anthropic/claude-haiku-4-5",
        "Claude Sonnet 4.6 (smart)": "anthropic/claude-sonnet-4-6",
        "GPT-4o Mini (cheap)":       "openai/gpt-4o-mini",
        "Llama 3.3 70B (free)":      "meta-llama/llama-3.3-70b-instruct:free",
    }
    selected_label = st.selectbox("Model", list(MODEL_OPTIONS.keys()))
    st.session_state["selected_model"] = MODEL_OPTIONS[selected_label]

    st.markdown("---")
    st.caption("M5 · Data Explorer")


# ════════════════════════════════════════════════════════════════
# MAIN AREA
# ════════════════════════════════════════════════════════════════
if uploaded_file is None:
    st.markdown('<p class="hero-title">🔍 Data Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">Upload a dataset in the sidebar to get started.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding: 4rem 0; color: #9ca3af;">
        <div style="font-size: 3.5rem;">📂</div>
        <div style="font-size: 1rem; margin-top: 0.8rem;">Use the sidebar to upload a CSV, JSON, or Excel file</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Parse file ────────────────────────────────────────────────
df_raw, error = parse_file(uploaded_file)
if error:
    st.error(error)
    st.stop()

# Large dataset sampling
df_display, was_sampled = maybe_sample(df_raw)
if was_sampled:
    st.warning(f"⚠ Dataset has {len(df_raw):,} rows — showing a random sample of {SAMPLE_THRESHOLD:,} for performance. Full data is used for exports.")

# Working copy stored in session_state (edits applied here)
edit_key = f"df_edit_{uploaded_file.name}"
if edit_key not in st.session_state:
    st.session_state[edit_key] = df_display.copy()

df = st.session_state[edit_key]
meta = describe_dataframe(df)

# AI client
client = get_ai_client(user_key=user_api_key)

# ── Summary cards (full width above split) ────────────────────
st.markdown(f"**{uploaded_file.name}**  ·  loaded")
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

st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)

# ── Two-column split ──────────────────────────────────────────
col_board, col_chat = st.columns([3, 2], gap="large")

# ════════════════════════════════════════════════════════════════
# LEFT — Analysis Board
# ════════════════════════════════════════════════════════════════
with col_board:
    st.markdown("### 📊 Analysis Board")

    # Pinned items from chat (rendered first so they accumulate at top)
    pinned_key = f"pinned_{uploaded_file.name}"
    if pinned_key not in st.session_state:
        st.session_state[pinned_key] = []

    pinned_items: list[dict] = st.session_state[pinned_key]

    # Data Overview
    with st.expander("📋 Column Overview", expanded=False):
        col_info = pd.DataFrame({
            "Column":   df.columns,
            "Type":     df.dtypes.astype(str).values,
            "Non-null": df.count().values,
            "Missing":  df.isnull().sum().values,
            "Sample":   [
                str(df[c].dropna().iloc[0]) if df[c].dropna().shape[0] > 0 else "(all null)"
                for c in df.columns
            ],
        })
        st.dataframe(col_info, use_container_width=True, hide_index=True)

    with st.expander("🗂 Raw Data Preview (first 100 rows)", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    all_num_cols = df.select_dtypes(include="number").columns.tolist()
    if all_num_cols:
        with st.expander("📊 Descriptive Statistics", expanded=False):
            st.caption("Columns likely to be IDs or flags are unchecked by default.")
            selected_cols = []
            cols_per_row = 3
            stat_rows = [all_num_cols[i:i+cols_per_row] for i in range(0, len(all_num_cols), cols_per_row)]
            for stat_row in stat_rows:
                cb_cols = st.columns(cols_per_row)
                for cb_col, col_name in zip(cb_cols, stat_row):
                    if cb_col.checkbox(col_name, value=is_meaningful_numeric(df[col_name]), key=f"stat_cb_{col_name}"):
                        selected_cols.append(col_name)
            if selected_cols:
                st.dataframe(df[selected_cols].describe().round(2), use_container_width=True)
            else:
                st.info("Check at least one column to see statistics.")

    with st.expander("🪪 Sample Record", expanded=False):
        best_row_idx = df.isnull().sum(axis=1).idxmin()
        sample = df.loc[best_row_idx]

        def is_nonempty(v) -> bool:
            try:
                if pd.isna(v):
                    return False
            except (ValueError, TypeError):
                pass
            if isinstance(v, (list, dict)) and len(v) == 0:
                return False
            return str(v).strip() != ""

        fields = [(k, v) for k, v in sample.items() if is_nonempty(v)]
        empty_fields = [(k, v) for k, v in sample.items() if not is_nonempty(v)]
        LONG_FIELD_THRESHOLD = 300

        for k, v in fields:
            val_str = str(v)
            is_long = len(val_str) > LONG_FIELD_THRESHOLD
            toggle_key = f"sample_expand_{k}"
            lbl, val = st.columns([1, 3])
            lbl.markdown(f'<div style="font-size:0.75rem;color:#6b7280;font-weight:600;padding:0.35rem 0;">{k}</div>', unsafe_allow_html=True)
            with val:
                if is_long:
                    exp = st.session_state.get(toggle_key, False)
                    disp = val_str if exp else val_str[:LONG_FIELD_THRESHOLD] + "…"
                    st.markdown(f'<div style="font-size:0.85rem;color:#111827;padding:0.35rem 0;white-space:pre-wrap;">{disp}</div>', unsafe_allow_html=True)
                    if st.button("Show less ▲" if exp else "Show more ▼", key=f"btn_{toggle_key}"):
                        st.session_state[toggle_key] = not exp
                        st.rerun()
                else:
                    st.markdown(f'<div style="font-size:0.85rem;color:#111827;padding:0.35rem 0;white-space:pre-wrap;">{val_str}</div>', unsafe_allow_html=True)
            st.divider()
        if empty_fields:
            st.caption(f"⚠ {len(empty_fields)} empty field(s): " + ", ".join(k for k, _ in empty_fields[:8]))

    # M4.5 Data Editing
    with st.expander("✏️ Data Editing", expanded=False):
        st.caption(f"Current: {len(df):,} rows × {len(df.columns)} columns")

        edit_tab1, edit_tab2, edit_tab3 = st.tabs(["🗑 Delete Rows", "➕ Add Row", "💾 Export"])

        with edit_tab1:
            st.markdown("**Delete rows manually**")
            del_indices = st.multiselect(
                "Select row indices to delete",
                options=df.index.tolist(),
                format_func=lambda i: f"Row {i}",
            )
            if st.button("Delete selected rows", disabled=len(del_indices) == 0):
                st.session_state[edit_key] = df.drop(index=del_indices).reset_index(drop=True)
                st.success(f"Deleted {len(del_indices)} row(s).")
                st.rerun()

            st.markdown("**Quick filters**")
            qf1, qf2 = st.columns(2)
            if qf1.button("🗑 Drop all rows with any null"):
                before = len(df)
                st.session_state[edit_key] = df.dropna().reset_index(drop=True)
                st.success(f"Removed {before - len(st.session_state[edit_key])} rows.")
                st.rerun()
            if qf2.button("🗑 Drop duplicate rows"):
                before = len(df)
                st.session_state[edit_key] = df.drop_duplicates().reset_index(drop=True)
                st.success(f"Removed {before - len(st.session_state[edit_key])} duplicates.")
                st.rerun()

            if st.button("↩ Reset all edits"):
                del st.session_state[edit_key]
                st.rerun()

        with edit_tab2:
            st.markdown("**Add a new row** (fill in values below)")
            new_row = {}
            for col_name in df.columns[:10]:  # limit to first 10 cols for usability
                new_row[col_name] = st.text_input(col_name, key=f"newrow_{col_name}", placeholder="(leave blank for null)")
            if st.columns([1, 3])[0].button("➕ Add row"):
                new_row_clean = {k: (v if v != "" else None) for k, v in new_row.items()}
                new_df = pd.concat([df, pd.DataFrame([new_row_clean])], ignore_index=True)
                st.session_state[edit_key] = new_df
                st.success("Row added.")
                st.rerun()

        with edit_tab3:
            st.markdown("**Export current dataset**")
            fmt = st.radio("Format", ["csv", "json", "excel"], horizontal=True)
            data_bytes, mime, ext = df_to_bytes(df, fmt)
            base_name = uploaded_file.name.rsplit(".", 1)[0]
            st.download_button(
                label=f"⬇ Download as .{ext}",
                data=data_bytes,
                file_name=f"{base_name}_edited.{ext}",
                mime=mime,
            )

    # AI Report
    st.markdown("---")
    st.markdown("#### 🤖 AI Report")
    if client is None:
        st.warning("⚠ Add an API key in the sidebar to enable AI features.")
    else:
        report_key = f"report_{uploaded_file.name}"
        if report_key not in st.session_state:
            with st.spinner("Generating report…"):
                try:
                    report = generate_report(client, build_analysis_prompt(uploaded_file.name, meta, df))
                    st.session_state[report_key] = report
                except Exception as e:
                    st.error(f"API call failed: {e}")
        else:
            st.markdown(f'<div class="report-box">{st.session_state[report_key]}</div>', unsafe_allow_html=True)
        if report_key in st.session_state:
            if st.button("🔄 Regenerate Report"):
                del st.session_state[report_key]
                st.rerun()

    # AI Charts
    st.markdown("---")
    st.markdown("#### 📈 AI-recommended Charts")
    if client is not None:
        charts_key = f"charts_{uploaded_file.name}"
        with st.spinner("Generating chart recommendations…"):
            specs = get_chart_specs(client, meta, df, charts_key)
        if specs:
            for i, spec in enumerate(specs):
                render_chart(spec, df, chart_key=f"auto_chart_{i}")
            if st.button("🔄 Refresh Charts"):
                del st.session_state[charts_key]
                st.rerun()

    # Pinned items from chat (at the bottom of the board)
    if pinned_items:
        st.markdown("---")
        st.markdown("#### 📌 Pinned from Chat")
        for i, item in enumerate(pinned_items):
            with st.expander(item.get("label", f"Item {i+1}"), expanded=True):
                if item["type"] == "fig":
                    st.plotly_chart(item["data"], use_container_width=True,
                                    key=f"pinned_fig_{i}")
                elif item["type"] == "df":
                    st.dataframe(item["data"], use_container_width=True)
            if st.button("🗑 Remove", key=f"unpin_{i}"):
                st.session_state[pinned_key].pop(i)
                st.rerun()


# ════════════════════════════════════════════════════════════════
# RIGHT — Chat
# ════════════════════════════════════════════════════════════════
with col_chat:
    st.markdown("### 💬 Chat")

    if client is None:
        st.warning("⚠ Add an API key in the sidebar to enable chat.")
    else:
        chat_key = f"chat_{uploaded_file.name}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []
        chat_history: list[dict] = st.session_state[chat_key]

        # Render existing messages
        for idx, msg in enumerate(chat_history):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "result_df" in msg:
                    st.dataframe(msg["result_df"], use_container_width=True)
                    if st.button("📌 Pin table", key=f"pin_df_{idx}"):
                        st.session_state[pinned_key].append({
                            "type": "df",
                            "data": msg["result_df"],
                            "label": f"Table · {msg['content'][:40]}…",
                        })
                        st.rerun()
                if "result_fig" in msg:
                    st.plotly_chart(msg["result_fig"], use_container_width=True,
                                    key=f"history_fig_{idx}")
                    if st.button("📌 Pin chart", key=f"pin_fig_{idx}"):
                        st.session_state[pinned_key].append({
                            "type": "fig",
                            "data": msg["result_fig"],
                            "label": f"Chart · {msg['content'][:40]}…",
                        })
                        st.rerun()

        # Chat input
        user_input = st.chat_input("Ask a question, request a chart, or edit your data…")

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            chat_history.append({"role": "user", "content": user_input})

            system_prompt = f"""You are a data analyst assistant. Always respond in English.

Dataset info:
- File: {uploaded_file.name}
- Shape: {meta['rows']} rows × {meta['columns']} columns
- Columns: {', '.join(meta['column_names'])}
- Numeric columns: {', '.join(meta['numeric_cols']) or 'none'}
- Text columns: {', '.join(meta['text_cols']) or 'none'}
- Missing rate: {meta['missing_pct']}

The DataFrame is available as `df`. Plotly Express is available as `px`.

You can help the user with three types of tasks:

1. DATA QUERIES — return a DataFrame/Series expression:
```python
df.nlargest(5, 'applicantsCount')[['title', 'companyName', 'applicantsCount']]
```

2. CHART REQUESTS — return a Plotly figure expression:
```python
px.bar(df.groupby('seniorityLevel')['applicantsCount'].mean().reset_index(), x='seniorityLevel', y='applicantsCount', title='Avg Applicants by Seniority')
```

3. DATA EDITING — return a DataFrame expression that produces the edited version:
```python
df.dropna(subset=['salary'])
```
For edits, start your explanation with "EDIT:" so the app knows to update the working dataset.

Rules:
- One expression per code block, no assignments, no imports, no file operations
- For conceptual questions, answer in plain text without code
- Always respond in English"""

            messages = [{"role": "system", "content": system_prompt}] + [
                {"role": m["role"], "content": m["content"]}
                for m in chat_history
            ]

            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                stream = client.chat.completions.create(
                    model=st.session_state.get("selected_model", "anthropic/claude-haiku-4-5"),
                    messages=messages,
                    stream=True,
                    max_tokens=1024,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full_response += delta
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)

                result_df = None
                result_fig = None

                if "```python" in full_response:
                    code_block = full_response.split("```python")[1].split("```")[0].strip()
                    BLOCKED = ["import", "open(", "write", "to_csv", "to_excel",
                               "to_json", "os.", "sys.", "eval(", "exec(",
                               "subprocess", "__import__"]
                    is_safe = not any(kw in code_block for kw in BLOCKED)

                    if is_safe:
                        try:
                            result = eval(code_block, {"df": df, "pd": pd, "px": px})
                            is_edit = "EDIT:" in full_response

                            if isinstance(result, go.Figure):
                                result_fig = result
                                chat_msg_idx = len(chat_history)  # index of message about to be stored
                                st.plotly_chart(result_fig, use_container_width=True,
                                                key=f"chat_fig_{chat_msg_idx}")
                                st.caption("Use the 📌 Pin chart button below to add this to the Analysis Board.")
                            elif isinstance(result, (pd.DataFrame, pd.Series)):
                                result_df = result if isinstance(result, pd.DataFrame) else result.to_frame()
                                if is_edit:
                                    st.session_state[edit_key] = result_df.reset_index(drop=True)
                                    st.success(f"✅ Dataset updated: now {len(result_df):,} rows × {len(result_df.columns)} columns.")
                                else:
                                    st.dataframe(result_df, use_container_width=True)
                                    st.caption("Use the 📌 Pin table button below to add this to the Analysis Board.")
                            else:
                                st.info(f"Result: {result}")
                        except Exception as e:
                            st.warning(f"Code execution failed: {e}")
                    else:
                        st.warning("Code blocked: contains disallowed operations.")

            assistant_msg: dict = {"role": "assistant", "content": full_response}
            if result_df is not None and "EDIT:" not in full_response:
                assistant_msg["result_df"] = result_df
            if result_fig is not None:
                assistant_msg["result_fig"] = result_fig
            chat_history.append(assistant_msg)
            # Rerun so pin buttons appear immediately on the stored message
            st.rerun()

        if chat_history:
            if st.button("🗑 Clear chat"):
                st.session_state[chat_key] = []
                st.rerun()
