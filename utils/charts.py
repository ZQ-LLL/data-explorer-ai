"""
utils/charts.py
Responsible for: rendering Plotly charts from AI-generated spec dicts.
Isolated here so chart logic can be extended or tested independently.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def render_chart(spec: dict, df: pd.DataFrame, chart_key: str = "") -> go.Figure | None:
    """
    Render a single Plotly chart from an AI spec dict.
    Supported types: bar, histogram, scatter, box, pie.
    Returns the Figure on success, None on failure.
    Skips gracefully if required columns are missing.
    """
    chart_type = spec.get("type", "").lower()
    title = spec.get("title", "Chart")
    x_col = spec.get("x")
    y_col = spec.get("y")
    color_col = spec.get("color")
    top_n = spec.get("top_n")
    reason = spec.get("reason", "")

    # Validate columns exist
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
            plot_bgcolor="white",
            paper_bgcolor="white",
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
