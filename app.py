"""
Data Explorer — app.py
Entry point: page config, styles, sidebar, and UI layout only.
All data logic lives in utils/parser.py, utils/ai.py, utils/charts.py.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from utils.ai import (
    build_analysis_prompt,
    build_column_analysis_prompt,
    build_chat_system_prompt,
    generate_report,
    get_ai_client,
    get_chart_specs,
)
from utils.agent import run_agent
from utils.charts import render_chart
from utils.parser import (
    SAMPLE_THRESHOLD,
    describe_dataframe,
    df_to_bytes,
    is_meaningful_numeric,
    maybe_sample,
    parse_file,
)

load_dotenv()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Data Explorer",
    page_icon="🔍",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────
st.markdown("""
<style>
    .stat-card {
        background: #f9fafb; border: 1px solid #e5e7eb;
        border-radius: 10px; padding: 0.75rem 1rem; text-align: center;
    }
    .stat-label {
        font-size: 0.7rem; color: #6b7280;
        text-transform: uppercase; letter-spacing: 0.05em;
    }
    .stat-value { font-size: 1.4rem; font-weight: 700; color: #111827; }
    .report-box {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
        padding: 1.2rem 1.5rem; font-size: 0.92rem; line-height: 1.75; color: #1e293b;
    }
    .chart-label { font-size: 0.78rem; color: #6b7280; font-style: italic; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 Data Explorer")
    st.markdown("---")
    st.markdown("### 📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        label="CSV · JSON · Excel",
        type=["csv", "json", "xlsx", "xls"],
        help="Recommended: under 200 MB",
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ⚙️ AI Settings")
    user_api_key = st.text_input(
        "API Key (OpenRouter)",
        type="password",
        placeholder="sk-or-v1-… (or set in .env)",
        help="Used for all AI features including Agent mode.",
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
# EMPTY STATE
# ════════════════════════════════════════════════════════════════
if uploaded_file is None:
    st.markdown('<p style="font-size:1.8rem;font-weight:700;">🔍 Data Explorer</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#6b7280;">Upload a dataset in the sidebar to get started.</p>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center;padding:4rem 0;color:#9ca3af;">
        <div style="font-size:3.5rem;">📂</div>
        <div style="font-size:1rem;margin-top:0.8rem;">Use the sidebar to upload a CSV, JSON, or Excel file</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════════════════════════════
# PARSE + SETUP
# ════════════════════════════════════════════════════════════════
df_raw, error = parse_file(uploaded_file)
if error:
    st.error(error)
    st.stop()

df_display, was_sampled = maybe_sample(df_raw)
if was_sampled:
    st.warning(
        f"⚠ Dataset has {len(df_raw):,} rows — showing a random sample of "
        f"{SAMPLE_THRESHOLD:,} for performance. Full data is used for exports."
    )

edit_key = f"df_edit_{uploaded_file.name}"
if edit_key not in st.session_state:
    st.session_state[edit_key] = df_display.copy()

df: pd.DataFrame = st.session_state[edit_key]
meta = describe_dataframe(df)
client = get_ai_client(user_key=user_api_key)
pinned_key = f"pinned_{uploaded_file.name}"
if pinned_key not in st.session_state:
    st.session_state[pinned_key] = []
pinned_items: list[dict] = st.session_state[pinned_key]


# ── Summary cards ─────────────────────────────────────────────
st.markdown(f"**{uploaded_file.name}**  ·  loaded")
c1, c2, c3, c4 = st.columns(4)
for col, label, value in [
    (c1, "Rows",         f"{meta['rows']:,}"),
    (c2, "Columns",      f"{meta['columns']}"),
    (c3, "Numeric cols", f"{len(meta['numeric_cols'])}"),
    (c4, "Missing",      meta["missing_pct"]),
]:
    col.markdown(
        f'<div class="stat-card"><div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
col_board, col_chat = st.columns([3, 2], gap="large")


# ════════════════════════════════════════════════════════════════
# LEFT — Analysis Board
# ════════════════════════════════════════════════════════════════
with col_board:
    st.markdown("### 📊 Analysis Board")

    # ── Column overview ───────────────────────────────────────
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

    # ── Raw preview ───────────────────────────────────────────
    with st.expander("🗂 Raw Data Preview (first 100 rows)", expanded=False):
        st.dataframe(df.head(100), use_container_width=True)

    # ── Descriptive statistics ────────────────────────────────
    all_num_cols = df.select_dtypes(include="number").columns.tolist()
    if all_num_cols:
        with st.expander("📊 Descriptive Statistics", expanded=False):
            st.caption("Columns likely to be IDs or flags are unchecked by default.")
            selected_cols = []
            for row in [all_num_cols[i:i+3] for i in range(0, len(all_num_cols), 3)]:
                for cb_col, col_name in zip(st.columns(3), row):
                    if cb_col.checkbox(col_name, value=is_meaningful_numeric(df[col_name]),
                                       key=f"stat_cb_{col_name}"):
                        selected_cols.append(col_name)
            if selected_cols:
                st.dataframe(df[selected_cols].describe().round(2), use_container_width=True)
            else:
                st.info("Check at least one column to see statistics.")

    # ── Sample record ─────────────────────────────────────────
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

        LONG = 300
        for k, v in [(k, v) for k, v in sample.items() if is_nonempty(v)]:
            val_str = str(v)
            toggle_key = f"sample_expand_{k}"
            lbl, val = st.columns([1, 3])
            lbl.markdown(f'<div style="font-size:0.75rem;color:#6b7280;font-weight:600;padding:0.35rem 0;">{k}</div>',
                         unsafe_allow_html=True)
            with val:
                if len(val_str) > LONG:
                    exp = st.session_state.get(toggle_key, False)
                    st.markdown(f'<div style="font-size:0.85rem;color:#111827;padding:0.35rem 0;white-space:pre-wrap;">'
                                f'{val_str if exp else val_str[:LONG] + "…"}</div>', unsafe_allow_html=True)
                    if st.button("Show less ▲" if exp else "Show more ▼", key=f"btn_{toggle_key}"):
                        st.session_state[toggle_key] = not exp
                        st.rerun()
                else:
                    st.markdown(f'<div style="font-size:0.85rem;color:#111827;padding:0.35rem 0;white-space:pre-wrap;">'
                                f'{val_str}</div>', unsafe_allow_html=True)
            st.divider()

    # ── Data editing ──────────────────────────────────────────
    with st.expander("✏️ Data Editing", expanded=False):
        st.caption(f"Current: {len(df):,} rows × {len(df.columns)} columns")
        edit_tab1, edit_tab2, edit_tab3 = st.tabs(["🗑 Delete Rows", "➕ Add Row", "💾 Export"])

        with edit_tab1:
            st.markdown("**Delete rows manually**")
            del_indices = st.multiselect("Select row indices to delete", options=df.index.tolist(),
                                         format_func=lambda i: f"Row {i}")
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
            st.markdown("**Add a new row**")
            new_row = {}
            for col_name in df.columns[:10]:
                new_row[col_name] = st.text_input(col_name, key=f"newrow_{col_name}",
                                                   placeholder="(leave blank for null)")
            if st.columns([1, 3])[0].button("➕ Add row"):
                new_row_clean = {k: (v if v != "" else None) for k, v in new_row.items()}
                st.session_state[edit_key] = pd.concat(
                    [df, pd.DataFrame([new_row_clean])], ignore_index=True
                )
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

    # ── Column analysis ───────────────────────────────────────
    st.markdown("---")
    with st.expander("🔬 Column Analysis", expanded=False):
        if client is None:
            st.warning("⚠ Add an API key in the sidebar to enable AI column analysis.")
        else:
            selected_col = st.selectbox("Select a column to analyse",
                                        options=df.columns.tolist(), key="col_analysis_select")
            if selected_col:
                series = df[selected_col]
                non_null = int(series.notna().sum())
                missing = len(series) - non_null
                try:
                    n_unique = int(series.nunique())
                    is_hashable = True
                except TypeError:
                    n_unique = -1
                    is_hashable = False

                s1, s2, s3, s4 = st.columns(4)
                s1.metric("Non-null", f"{non_null:,}")
                s2.metric("Missing", f"{missing:,} ({round(missing/len(series)*100,1)}%)")
                s3.metric("Unique values", f"{n_unique:,}" if is_hashable else "N/A (nested)")
                s4.metric("Type", str(series.dtype))

                if not is_hashable:
                    st.caption("⚠ This column contains nested objects. Showing samples:")
                    for s in series.dropna().head(5).tolist():
                        st.markdown(
                            f'<div style="font-size:0.83rem;background:#f8fafc;border-left:3px solid '
                            f'#cbd5e1;padding:0.4rem 0.7rem;margin-bottom:0.3rem;border-radius:0 6px 6px 0;">'
                            f'{str(s)[:300]}</div>', unsafe_allow_html=True)
                elif pd.api.types.is_numeric_dtype(series):
                    import plotly.express as _px
                    fig = _px.histogram(series.dropna().to_frame(), x=selected_col,
                                        title=f"Distribution of {selected_col}", height=250)
                    fig.update_layout(margin=dict(t=35,b=25,l=25,r=10),
                                      plot_bgcolor="white", paper_bgcolor="white")
                    st.plotly_chart(fig, use_container_width=True, key=f"col_hist_{selected_col}")
                elif n_unique <= 30:
                    top_vals = series.value_counts().head(10).reset_index()
                    top_vals.columns = [selected_col, "count"]
                    import plotly.express as _px
                    fig = _px.bar(top_vals, x=selected_col, y="count",
                                  title=f"Top values in {selected_col}", height=250)
                    fig.update_layout(margin=dict(t=35,b=25,l=25,r=10),
                                      plot_bgcolor="white", paper_bgcolor="white")
                    st.plotly_chart(fig, use_container_width=True, key=f"col_bar_{selected_col}")
                else:
                    st.caption("Sample values (text column):")
                    for s in series.dropna().sample(min(5, non_null), random_state=42).tolist():
                        st.markdown(
                            f'<div style="font-size:0.83rem;background:#f8fafc;border-left:3px solid '
                            f'#cbd5e1;padding:0.4rem 0.7rem;margin-bottom:0.3rem;border-radius:0 6px 6px 0;">'
                            f'{str(s)[:300]}</div>', unsafe_allow_html=True)

                col_analysis_key = f"col_analysis_{uploaded_file.name}_{selected_col}"
                if st.button("🤖 Analyse this column with AI", key=f"btn_analyse_{selected_col}"):
                    with st.spinner(f"Analysing '{selected_col}'…"):
                        try:
                            prompt = build_column_analysis_prompt(selected_col, series, df)
                            response = client.chat.completions.create(
                                model=st.session_state.get("selected_model", "anthropic/claude-haiku-4-5"),
                                messages=[{"role": "user", "content": prompt}],
                                stream=False, max_tokens=800,
                            )
                            st.session_state[col_analysis_key] = response.choices[0].message.content.strip()
                        except Exception as e:
                            st.error(f"Analysis failed: {e}")
                if col_analysis_key in st.session_state:
                    st.markdown("**AI Analysis:**")
                    st.markdown(f'<div class="report-box">{st.session_state[col_analysis_key]}</div>',
                                unsafe_allow_html=True)
                    if st.button("🗑 Clear analysis", key=f"clear_col_{selected_col}"):
                        del st.session_state[col_analysis_key]
                        st.rerun()

    # ── AI report ─────────────────────────────────────────────
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
            st.markdown(f'<div class="report-box">{st.session_state[report_key]}</div>',
                        unsafe_allow_html=True)
        if report_key in st.session_state:
            if st.button("🔄 Regenerate Report"):
                del st.session_state[report_key]
                st.rerun()

    # ── AI charts ─────────────────────────────────────────────
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

    # ── Pinned items ──────────────────────────────────────────
    if pinned_items:
        st.markdown("---")
        st.markdown("#### 📌 Pinned from Chat")
        for i, item in enumerate(pinned_items):
            with st.expander(item.get("label", f"Item {i+1}"), expanded=True):
                if item["type"] == "fig":
                    st.plotly_chart(item["data"], use_container_width=True, key=f"pinned_fig_{i}")
                elif item["type"] == "df":
                    st.dataframe(item["data"], use_container_width=True)
            if st.button("🗑 Remove", key=f"unpin_{i}"):
                st.session_state[pinned_key].pop(i)
                st.rerun()


# ════════════════════════════════════════════════════════════════
# RIGHT — Chat + Agent
# ════════════════════════════════════════════════════════════════
with col_chat:
    chat_tab, agent_tab = st.tabs(["💬 Chat", "🤖 Agent"])

    # Initialise chat state outside tabs so agent tab can also access it
    chat_key = f"chat_{uploaded_file.name}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []
    chat_history: list[dict] = st.session_state[chat_key]

    # ════════════════════════════════════════════════════════════
    # CHAT TAB
    # ════════════════════════════════════════════════════════════
    with chat_tab:
        if client is None:
            st.warning("⚠ Add an OpenRouter API key in the sidebar to enable chat.")
        else:
            # Render existing messages
            for idx, msg in enumerate(chat_history):
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if "result_df" in msg:
                        st.dataframe(msg["result_df"], use_container_width=True)
                        if st.button("📌 Pin table", key=f"pin_df_{idx}"):
                            st.session_state[pinned_key].append({
                                "type": "df", "data": msg["result_df"],
                                "label": f"Table · {msg['content'][:40]}…",
                            })
                            st.rerun()
                    if "result_fig" in msg:
                        st.plotly_chart(msg["result_fig"], use_container_width=True,
                                        key=f"history_fig_{idx}")
                        if st.button("📌 Pin chart", key=f"pin_fig_{idx}"):
                            st.session_state[pinned_key].append({
                                "type": "fig", "data": msg["result_fig"],
                                "label": f"Chart · {msg['content'][:40]}…",
                            })
                            st.rerun()

            # chat_input inside chat_tab — only renders when this tab is active
            user_input = st.chat_input(
                "Ask a question, request a chart, or edit your data…",
                key="chat_input",
            )

            # Handle new input
            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                chat_history.append({"role": "user", "content": user_input})

                messages = [{"role": "system", "content": build_chat_system_prompt(uploaded_file.name, meta)}] + [
                    {"role": m["role"], "content": m["content"]} for m in chat_history
                ]

                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    full_response = ""
                    stream = client.chat.completions.create(
                        model=st.session_state.get("selected_model", "anthropic/claude-haiku-4-5"),
                        messages=messages, stream=True, max_tokens=1024,
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        full_response += delta
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)

                    result_df = None
                    result_fig = None

                    if "```python" in full_response:
                        code_block = full_response.split("```python")[1].split("```")[0].strip()
                        BLOCKED = ["import", "open(", "write", "to_csv", "to_excel",
                                   "to_json", "os.", "sys.", "eval(", "exec(",
                                   "subprocess", "__import__"]
                        if not any(kw in code_block for kw in BLOCKED):
                            try:
                                result = eval(code_block, {"df": df, "pd": pd, "px": px})
                                is_edit = "EDIT:" in full_response
                                if isinstance(result, go.Figure):
                                    result_fig = result
                                    st.plotly_chart(result_fig, use_container_width=True,
                                                    key=f"chat_fig_{len(chat_history)}")
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
                st.rerun()

            if chat_history:
                if st.button("🗑 Clear chat", key="clear_chat"):
                    st.session_state[chat_key] = []
                    st.rerun()

    # ════════════════════════════════════════════════════════════
    # AGENT TAB
    # ════════════════════════════════════════════════════════════
    with agent_tab:
        st.caption("Agent plans and executes multiple analysis steps autonomously.")

        import os
        resolved_key = user_api_key.strip() or os.getenv("OPENROUTER_API_KEY", "").strip()

        if not resolved_key:
            st.warning("⚠ Add an OpenRouter API key in the sidebar to use Agent mode.")
        else:
            AGENT_MODELS = {
                "claude-haiku-4-5 (fast, cheap)": "anthropic/claude-haiku-4-5",
                "claude-sonnet-4-6 (smart)":      "anthropic/claude-sonnet-4-6",
            }
            agent_model_label = st.selectbox(
                "Agent model",
                list(AGENT_MODELS.keys()),
                key="agent_model_select",
            )
            agent_model = AGENT_MODELS[agent_model_label]

            # Example goals to help the user get started
            with st.expander("💡 Example goals", expanded=False):
                st.markdown("""
- Give me a complete overview of this dataset
- Find the columns with the most missing data and explain what that means
- Show me the distribution of key numeric columns
- What are the top 10 most common values in each categorical column?
- Clean the data by removing rows where seniority level is missing, then summarize what's left
                """)

            user_goal = st.text_area(
                "What do you want the Agent to analyse?",
                placeholder="e.g. Give me a complete overview of this dataset and highlight the most interesting patterns.",
                height=100,
                key="agent_goal_input",
            )

            agent_events_key = f"agent_events_{uploaded_file.name}"

            col_run, col_clear = st.columns([2, 1])
            run_clicked = col_run.button("▶ Run Agent", type="primary",
                                         disabled=not user_goal.strip())
            if col_clear.button("🗑 Clear", key="clear_agent"):
                st.session_state.pop(agent_events_key, None)
                st.rerun()

            if run_clicked and user_goal.strip():
                st.session_state.pop(agent_events_key, None)
                with st.spinner("Agent is working…"):
                    events = run_agent(
                        user_goal=user_goal,
                        df=df,
                        openrouter_api_key=resolved_key,
                        model=agent_model,
                        max_steps=10,
                    )
                st.session_state[agent_events_key] = events

                # Apply any dataset updates from the agent
                for ev in events:
                    if ev["type"] == "df_update":
                        st.session_state[edit_key] = ev["df"]
                    if ev["type"] == "_final_df":
                        # Update df in session if it changed
                        if len(ev["df"]) != len(df):
                            st.session_state[edit_key] = ev["df"]

                st.rerun()

            # ── Render stored agent events ────────────────────
            if agent_events_key in st.session_state:
                events = st.session_state[agent_events_key]
                step_num = 0

                for ev in events:
                    if ev["type"] == "_final_df":
                        continue  # internal use only

                    elif ev["type"] == "text":
                        st.markdown(ev["content"])

                    elif ev["type"] == "tool":
                        step_num += 1
                        with st.expander(f"🔧 Step {step_num}: {ev['name']}", expanded=False):
                            if ev["input"]:
                                for k, v in ev["input"].items():
                                    st.caption(f"**{k}:** {str(v)[:200]}")

                    elif ev["type"] == "result":
                        st.code(ev["content"], language=None)

                    elif ev["type"] == "figure":
                        st.plotly_chart(ev["fig"], use_container_width=True,
                                        key=f"agent_fig_{step_num}_{ev['title'][:20]}")
                        if st.button("📌 Pin to Board", key=f"pin_agent_fig_{step_num}"):
                            st.session_state[pinned_key].append({
                                "type": "fig",
                                "data": ev["fig"],
                                "label": f"Agent · {ev['title']}",
                            })
                            st.rerun()

                    elif ev["type"] == "df_update":
                        st.success(f"✅ {ev['message']}")

                    elif ev["type"] == "summary":
                        st.markdown("---")
                        st.markdown("### 📋 Summary")
                        st.markdown(f'<div class="report-box">{ev["content"]}</div>',
                                    unsafe_allow_html=True)

                    elif ev["type"] == "error":
                        st.error(ev["content"])

                    elif ev["type"] == "limit":
                        st.warning(ev["content"])
