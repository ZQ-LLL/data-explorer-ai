"""
Data Explorer — M3: Conversational Multi-turn Chat + AI Code Execution
"""

import io
import os

import pandas as pd
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
    .report-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        font-size: 0.95rem;
        line-height: 1.75;
        color: #1e293b;
    }
</style>
""", unsafe_allow_html=True)


# ── OpenRouter client ─────────────────────────────────────────
@st.cache_resource
def get_ai_client() -> OpenAI | None:
    """
    Initialize OpenRouter client using the API key from .env.
    Returns None if the key is missing.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


# ── Parsing ───────────────────────────────────────────────────
def parse_file(uploaded_file) -> tuple[pd.DataFrame | None, str]:
    """
    Parse an uploaded file based on its extension.
    Returns (DataFrame, error_message). On success, error_message is empty.
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
    Excludes ID-like columns and boolean-encoded columns (only 0/1 values).
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
    Extract key metadata from a DataFrame.
    Stored in session_state and reused by the M2 AI prompt builder.
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


# ── AI prompt builder ─────────────────────────────────────────
def build_analysis_prompt(filename: str, meta: dict, df: pd.DataFrame) -> str:
    """
    Build a structured prompt for the AI report.
    Includes dataset metadata and sample statistics so the AI has concrete numbers to work with.
    """
    # Build a short stats summary for meaningful numeric columns
    stats_lines = []
    if meta["numeric_cols"]:
        stats = df[meta["numeric_cols"]].describe().round(2)
        for col in meta["numeric_cols"]:
            s = stats[col]
            stats_lines.append(
                f"  - {col}: mean={s['mean']}, min={s['min']}, max={s['max']}, "
                f"missing={df[col].isnull().sum()}"
            )

    # Top columns by missing value count (potential data quality signals)
    missing_by_col = df.isnull().sum().sort_values(ascending=False)
    top_missing = missing_by_col[missing_by_col > 0].head(8)
    missing_lines = [f"  - {col}: {cnt} missing" for col, cnt in top_missing.items()]

    prompt = f"""You are a data analyst. A user has uploaded a dataset and needs a clear, insightful report.

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

Please write a concise dataset analysis report (around 200-300 words) that covers:
1. What this dataset likely contains and its probable purpose
2. Key observations about the data structure and quality
3. Notable patterns or potential issues worth investigating
4. 2-3 specific analysis questions this dataset could answer

Write in clear, plain English. Be specific — reference actual column names and numbers. Avoid generic filler phrases.
"""
    return prompt


# ── AI report generator ───────────────────────────────────────
def generate_report(client: OpenAI, prompt: str) -> str:
    """
    Call the OpenRouter API and return the full response text.
    Uses streaming so the user sees text appear progressively.
    """
    stream = client.chat.completions.create(
        model="anthropic/claude-haiku-4-5",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        max_tokens=1024,
    )

    report_placeholder = st.empty()
    full_text = ""
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        full_text += delta
        report_placeholder.markdown(
            f'<div class="report-box">{full_text}▌</div>',
            unsafe_allow_html=True,
        )

    # Final render without the blinking cursor
    report_placeholder.markdown(
        f'<div class="report-box">{full_text}</div>',
        unsafe_allow_html=True,
    )
    return full_text


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

    # Store in session_state so M3 can access without re-parsing
    st.session_state["df"] = df
    st.session_state["filename"] = uploaded_file.name

    meta = describe_dataframe(df)
    st.session_state["meta"] = meta

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

    # ── Raw data preview ──────────────────────────────────────
    with st.expander("🗂 Raw Data Preview (first 100 rows)", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    # ── Descriptive statistics with interactive column selector ──
    all_num_cols = df.select_dtypes(include="number").columns.tolist()
    if all_num_cols:
        with st.expander("📊 Descriptive Statistics", expanded=False):
            st.caption("Select which numeric columns to include. Columns likely to be IDs or flags are unchecked by default.")
            selected_cols = []
            cols_per_row = 4
            rows = [all_num_cols[i:i+cols_per_row] for i in range(0, len(all_num_cols), cols_per_row)]
            for row in rows:
                cb_cols = st.columns(cols_per_row)
                for cb_col, col_name in zip(cb_cols, row):
                    default = is_meaningful_numeric(df[col_name])
                    if cb_col.checkbox(col_name, value=default, key=f"stat_cb_{col_name}"):
                        selected_cols.append(col_name)
            if selected_cols:
                st.dataframe(df[selected_cols].describe().round(2), use_container_width=True)
            else:
                st.info("Check at least one column above to see statistics.")

    # ── Sample record ─────────────────────────────────────────
    with st.expander("🪪 Sample Record (first non-null row)", expanded=True):
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

        # Threshold above which a value gets a "show more" toggle
        LONG_FIELD_THRESHOLD = 300

        if fields:
            for k, v in fields:
                val_str = str(v)
                is_long = len(val_str) > LONG_FIELD_THRESHOLD
                toggle_key = f"sample_expand_{k}"

                col_label, col_value = st.columns([1, 3])
                col_label.markdown(
                    f'<div style="font-size:0.78rem;color:#6b7280;font-weight:600;'
                    f'padding:0.4rem 0;word-break:break-word;">{k}</div>',
                    unsafe_allow_html=True,
                )
                with col_value:
                    if is_long:
                        expanded = st.session_state.get(toggle_key, False)
                        display_val = val_str if expanded else val_str[:LONG_FIELD_THRESHOLD] + "…"
                        st.markdown(
                            f'<div style="font-size:0.88rem;color:#111827;padding:0.4rem 0;'
                            f'word-break:break-word;white-space:pre-wrap;">{display_val}</div>',
                            unsafe_allow_html=True,
                        )
                        btn_label = "Show less ▲" if expanded else "Show more ▼"
                        if st.button(btn_label, key=f"btn_{toggle_key}"):
                            st.session_state[toggle_key] = not expanded
                            st.rerun()
                    else:
                        st.markdown(
                            f'<div style="font-size:0.88rem;color:#111827;padding:0.4rem 0;'
                            f'word-break:break-word;white-space:pre-wrap;">{val_str}</div>',
                            unsafe_allow_html=True,
                        )
                st.divider()

        if empty_fields:
            st.caption(
                f"⚠ {len(empty_fields)} field(s) are empty in this record: "
                + ", ".join(k for k, _ in empty_fields[:10])
                + ("…" if len(empty_fields) > 10 else "")
            )

    # ── M2: AI Analysis Report ────────────────────────────────
    st.markdown("---")
    st.subheader("🤖 AI Analysis Report")

    client = get_ai_client()

    if client is None:
        st.warning("⚠ No API key found. Add OPENROUTER_API_KEY to your .env file to enable AI reports.")
    else:
        # Only auto-generate once per file; store report in session_state
        report_key = f"report_{uploaded_file.name}"

        if report_key not in st.session_state:
            with st.spinner("Analyzing your dataset…"):
                prompt = build_analysis_prompt(uploaded_file.name, meta, df)
                try:
                    report = generate_report(client, prompt)
                    st.session_state[report_key] = report
                except Exception as e:
                    st.error(f"API call failed: {e}")
        else:
            # Show cached report with regenerate option
            st.markdown(
                f'<div class="report-box">{st.session_state[report_key]}</div>',
                unsafe_allow_html=True,
            )

        if report_key in st.session_state:
            if st.button("🔄 Regenerate Report"):
                del st.session_state[report_key]
                st.rerun()

    # ── M3: Conversational Chat ───────────────────────────────
    st.markdown("---")
    st.subheader("💬 Ask About Your Data")

    if client is None:
        st.warning("⚠ No API key found. Add OPENROUTER_API_KEY to your .env file to enable chat.")
    else:
        # Initialise chat history for this file
        chat_key = f"chat_{uploaded_file.name}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        chat_history: list[dict] = st.session_state[chat_key]

        # ── Render existing messages ──────────────────────────
        for msg in chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Re-render any dataframe result stored with the message
                if "result_df" in msg:
                    st.dataframe(msg["result_df"], use_container_width=True)

        # ── Chat input ────────────────────────────────────────
        user_input = st.chat_input("Ask a question about your dataset…")

        if user_input:
            # Show and store user message
            with st.chat_message("user"):
                st.markdown(user_input)
            chat_history.append({"role": "user", "content": user_input})

            # ── Build system prompt ───────────────────────────
            system_prompt = f"""You are a data analyst assistant. The user has uploaded a dataset and wants to explore it through conversation.

Dataset info:
- File: {uploaded_file.name}
- Shape: {meta['rows']} rows × {meta['columns']} columns
- Columns: {', '.join(meta['column_names'])}
- Numeric columns: {', '.join(meta['numeric_cols']) or 'none'}
- Text columns: {', '.join(meta['text_cols']) or 'none'}
- Missing rate: {meta['missing_pct']}

The DataFrame is available as the variable `df`.

When answering questions:
1. Write a brief, clear explanation of your findings.
2. If the question can be answered more precisely with data, include a code block with a single Python expression using `df` that returns a DataFrame or Series. Wrap it in ```python ... ```.
3. Keep code simple — one expression, no assignments, no imports, no file operations.
4. If the question is conceptual or interpretive, just answer in plain text without code.

Example of good code:
```python
df.nlargest(5, 'applicantsCount')[['title', 'companyName', 'applicantsCount']]
```"""

            # Build messages list: system + full history
            messages = [{"role": "system", "content": system_prompt}] + [
                {"role": m["role"], "content": m["content"]}
                for m in chat_history
            ]

            # ── Call AI and stream response ───────────────────
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""

                stream = client.chat.completions.create(
                    model="anthropic/claude-haiku-4-5",
                    messages=messages,
                    stream=True,
                    max_tokens=1024,
                )
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    full_response += delta
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)

                # ── Extract and execute code block if present ─
                result_df = None
                if "```python" in full_response:
                    code_block = full_response.split("```python")[1].split("```")[0].strip()

                    # Safety check: block any write/import operations
                    BLOCKED_KEYWORDS = [
                        "import", "open(", "write", "to_csv", "to_excel",
                        "to_json", "os.", "sys.", "eval(", "exec(",
                        "subprocess", "__import__",
                    ]
                    is_safe = not any(kw in code_block for kw in BLOCKED_KEYWORDS)

                    if is_safe:
                        try:
                            result = eval(code_block, {"df": df, "pd": pd})
                            if isinstance(result, (pd.DataFrame, pd.Series)):
                                result_df = result if isinstance(result, pd.DataFrame) else result.to_frame()
                                st.dataframe(result_df, use_container_width=True)
                            else:
                                # Scalar result (count, mean, etc.)
                                st.info(f"Result: {result}")
                        except Exception as e:
                            st.warning(f"Code execution failed: {e}")
                    else:
                        st.warning("Code blocked: contains disallowed operations.")

            # Store assistant message (and result df if any)
            assistant_msg: dict = {"role": "assistant", "content": full_response}
            if result_df is not None:
                assistant_msg["result_df"] = result_df
            chat_history.append(assistant_msg)

        # ── Clear chat button ─────────────────────────────────
        if chat_history:
            if st.button("🗑 Clear conversation"):
                st.session_state[chat_key] = []
                st.rerun()

    # ── M4 handoff notice ─────────────────────────────────────
    st.markdown("""
    <div class="info-box" style="margin-top:1rem;">
    ✨ <b>M3 complete</b> — Conversational chat with code execution enabled.
    Next up (M4): AI-recommended dynamic Plotly visualizations.
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
