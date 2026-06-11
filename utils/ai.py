"""
utils/ai.py
Responsible for: AI client setup, prompt building, API calls.
Streamlit is imported only for streaming output (st.empty).
All prompt builders are pure functions — testable without a running app.
"""

import json
import os

import pandas as pd
import streamlit as st
from openai import OpenAI


def get_ai_client(user_key: str = "") -> OpenAI | None:
    """
    Build an OpenAI-compatible client pointed at OpenRouter.
    User-provided key takes priority over the .env OPENROUTER_API_KEY.
    Returns None if no key is available.
    """
    api_key = user_key.strip() or os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
    )


# ── Prompt builders ───────────────────────────────────────────
# Each function takes plain Python data (dicts, DataFrames, strings)
# and returns a plain string. No side effects, easy to test or tweak.

def build_analysis_prompt(filename: str, meta: dict, df: pd.DataFrame) -> str:
    """Build the dataset-level analysis report prompt."""
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
    """Build the chart recommendation prompt, requesting a JSON array response."""
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


def build_column_analysis_prompt(col_name: str, series: pd.Series, df: pd.DataFrame) -> str:
    """
    Build a deep-dive prompt for a single column.
    Adapts content based on column type: numeric, categorical, text, or nested object.
    """
    dtype = str(series.dtype)
    total = len(series)
    non_null = int(series.notna().sum())
    missing = total - non_null
    missing_pct = round(missing / total * 100, 1)

    try:
        n_unique = int(series.nunique())
        is_hashable = True
    except TypeError:
        n_unique = -1
        is_hashable = False

    if pd.api.types.is_numeric_dtype(series):
        col_type = "numeric"
        stats = series.describe().round(3)
        extra = f"""Statistics:
  mean={stats['mean']}, median={series.median():.3f}, std={stats['std']:.3f}
  min={stats['min']}, max={stats['max']}
  25th pct={stats['25%']}, 75th pct={stats['75%']}
  skewness={series.skew():.3f}"""

    elif not is_hashable:
        col_type = "nested object (dict/list)"
        samples = series.dropna().head(5).tolist()
        extra = "This column contains nested objects. Sample values:\n" + \
                "\n".join(f"  - {str(v)[:200]}" for v in samples)

    elif n_unique <= 30:
        col_type = "categorical"
        top_vals = series.value_counts().head(10)
        extra = "Top values (count):\n" + \
                "\n".join(f"  {v}: {c}" for v, c in top_vals.items())

    else:
        col_type = "free text"
        sample_vals = series.dropna().sample(min(5, non_null), random_state=42).tolist()
        avg_len = int(series.dropna().astype(str).str.len().mean())
        extra = f"Average length: {avg_len} characters\nSample values:\n" + \
                "\n".join(f"  - {str(v)[:200]}" for v in sample_vals)

    return f"""You are a data analyst. Perform a deep analysis of a single column from a dataset.

Column: "{col_name}"
Type: {col_type} ({dtype})
Total rows: {total}
Non-null: {non_null} ({100 - missing_pct:.1f}%)
Missing: {missing} ({missing_pct}%)
Unique values: {n_unique if is_hashable else 'N/A (nested objects)'}

{extra}

Please provide a focused analysis (150-250 words) covering:
1. What this column likely represents and its role in the dataset
2. Key patterns, distributions, or anomalies in the data
3. Data quality observations (missing values, outliers, inconsistencies)
4. 1-2 specific insights or follow-up questions this column raises

Be specific and reference actual values/numbers. Always respond in English.
"""


def build_chat_system_prompt(filename: str, meta: dict) -> str:
    """Build the system prompt injected at the start of every chat API call."""
    return f"""You are a data analyst assistant. Always respond in English.

Dataset info:
- File: {filename}
- Shape: {meta['rows']} rows × {meta['columns']} columns
- Columns: {', '.join(meta['column_names'])}
- Numeric columns: {', '.join(meta['numeric_cols']) or 'none'}
- Text columns: {', '.join(meta['text_cols']) or 'none'}
- Missing rate: {meta['missing_pct']}

The DataFrame is available as `df`. Plotly Express is available as `px`.

You can help with three types of tasks:

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


# ── API callers ───────────────────────────────────────────────

def generate_report(client: OpenAI, prompt: str) -> str:
    """
    Stream the AI report into a Streamlit placeholder.
    Returns the full response text when done.
    """
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


def get_chart_specs(client: OpenAI, meta: dict, df: pd.DataFrame, cache_key: str) -> list[dict] | None:
    """
    Request chart recommendations from AI as a JSON array.
    Caches result in session_state so it survives reruns without re-calling the API.
    """
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
