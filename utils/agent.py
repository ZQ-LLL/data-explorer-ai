"""
utils/agent.py
Responsible for: Agent loop, tool definitions, tool execution.

Uses OpenRouter via the OpenAI-compatible SDK (same key as the rest of the app).
Tool calling format follows the OpenAI function-calling standard, which OpenRouter
supports for Claude models.

Architecture:
  - TOOLS: list of tool schemas that tell Claude what it can do
  - execute_tool(): runs the actual Python code for each tool
  - run_agent(): the main loop — calls API, handles tool_use, repeats until done
"""

import json
import traceback

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

# ── Tool schemas ──────────────────────────────────────────────
# OpenRouter uses the OpenAI function-calling format:
# each tool has a "type": "function" wrapper around "function" with
# name, description, and parameters (JSON Schema).
#
# The "description" field is critical — Claude reads it to decide
# whether to use this tool for a given situation.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "inspect_dataframe",
            "description": (
                "Get a structural overview of the dataset: column names, data types, "
                "non-null counts, missing rates, and sample values. "
                "Use this as your FIRST step whenever you need to understand what "
                "the dataset contains before deciding what analysis to run."
            ),
            "parameters": {
                "type": "object",
                "properties": {},       # no parameters — df is always in scope
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_column_stats",
            "description": (
                "Get detailed statistics for a specific column: value counts (categorical), "
                "mean/std/min/max (numeric), or sample values (free text / nested). "
                "Use this when you need to understand a particular column in depth "
                "before drawing conclusions or creating a chart."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "column_name": {
                        "type": "string",
                        "description": "The exact name of the column to analyse. Must exist in the dataset.",
                    }
                },
                "required": ["column_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_pandas_query",
            "description": (
                "Execute a single Pandas expression on the DataFrame (variable: `df`) "
                "and return the result as text. "
                "Use this for precise data queries: filtering rows, aggregating values, "
                "finding top/bottom N, computing correlations, counting, etc. "
                "Returns up to 20 rows. "
                "Do NOT use for chart creation — use create_chart instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A single valid Pandas expression using `df`. "
                            "Examples: "
                            "df.nlargest(5, 'applicantsCount')[['title', 'companyName']]  "
                            "or  df['seniorityLevel'].value_counts()"
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_chart",
            "description": (
                "Create a Plotly chart and display it in the UI. "
                "Use this when the user asks for a visualisation, or when a chart "
                "communicates a pattern better than a table. "
                "Supported types: bar, histogram, scatter, box, pie."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "chart_type": {
                        "type": "string",
                        "enum": ["bar", "histogram", "scatter", "box", "pie"],
                        "description": "The type of chart to create.",
                    },
                    "x": {
                        "type": "string",
                        "description": "Column name for the x-axis (or names for pie chart).",
                    },
                    "y": {
                        "type": "string",
                        "description": (
                            "Column name for the y-axis. "
                            "Use 'count' for bar/pie charts that count occurrences. "
                            "Not required for histogram."
                        ),
                    },
                    "title": {
                        "type": "string",
                        "description": "Chart title.",
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "Limit to top N categories (useful for crowded bar charts).",
                    },
                },
                "required": ["chart_type", "x", "title"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "filter_dataframe",
            "description": (
                "Apply a filter or transformation to the working dataset and update it permanently. "
                "Use ONLY when the user explicitly asks to clean, filter, or edit the data "
                "(e.g. 'remove rows with missing salary', 'drop duplicates'). "
                "The result becomes the new working dataset for all subsequent steps."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "A Pandas expression using `df` that returns a filtered DataFrame. "
                            "Examples: df.dropna(subset=['salary'])  "
                            "or  df[df['seniorityLevel'] != 'Not Applicable']"
                        ),
                    },
                    "description": {
                        "type": "string",
                        "description": "Short human-readable description of what this filter does.",
                    },
                },
                "required": ["expression", "description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_findings",
            "description": (
                "Write a final summary of all analysis findings from this session. "
                "Always call this as your LAST step to give the user a clear conclusion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "A clear, structured summary of everything discovered.",
                    }
                },
                "required": ["summary"],
            },
        },
    },
]

# Keywords never allowed in executed code
BLOCKED_KEYWORDS = [
    "import", "open(", "write", "to_csv", "to_excel",
    "to_json", "os.", "sys.", "eval(", "exec(",
    "subprocess", "__import__",
]


# ── Tool execution ────────────────────────────────────────────

def execute_tool(
    tool_name: str,
    tool_input: dict,
    df: pd.DataFrame,
) -> tuple[str, go.Figure | None, pd.DataFrame | None]:
    """
    Execute a tool and return (text_result, optional_figure, optional_new_df).

    text_result  : always returned — goes back to the model as the tool result
    figure       : Plotly figure if a chart was created, else None
    updated_df   : new DataFrame if the dataset was modified, else None
    """

    # ── inspect_dataframe ─────────────────────────────────────
    if tool_name == "inspect_dataframe":
        lines = [f"Shape: {len(df):,} rows × {len(df.columns)} columns\n"]
        lines.append(f"{'Column':<30} {'Type':<12} {'Non-null':>8} {'Missing':>8}  Sample")
        lines.append("-" * 80)
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null = int(df[col].notna().sum())
            missing = len(df) - non_null
            try:
                sample = str(df[col].dropna().iloc[0])[:40] if non_null > 0 else "(all null)"
            except Exception:
                sample = "(error)"
            lines.append(f"{col:<30} {dtype:<12} {non_null:>8} {missing:>8}  {sample}")
        return "\n".join(lines), None, None

    # ── get_column_stats ──────────────────────────────────────
    elif tool_name == "get_column_stats":
        col_name = tool_input["column_name"]
        if col_name not in df.columns:
            return (
                f"Error: column '{col_name}' not found. "
                f"Available columns: {', '.join(df.columns[:20])}",
                None, None,
            )
        series = df[col_name]
        non_null = int(series.notna().sum())
        missing = len(series) - non_null
        lines = [
            f"Column: {col_name}",
            f"Type: {series.dtype}",
            f"Non-null: {non_null:,} ({round(non_null/len(series)*100,1)}%)",
            f"Missing: {missing:,} ({round(missing/len(series)*100,1)}%)",
        ]
        if pd.api.types.is_numeric_dtype(series):
            stats = series.describe().round(3)
            lines += [
                f"Mean: {stats['mean']}  |  Std: {stats['std']}",
                f"Min: {stats['min']}  |  Max: {stats['max']}",
                f"25th / 50th / 75th: {stats['25%']} / {stats['50%']} / {stats['75%']}",
            ]
        else:
            try:
                n_unique = series.nunique()
                lines.append(f"Unique values: {n_unique:,}")
                top = series.value_counts().head(10)
                lines.append("Top values:")
                for val, cnt in top.items():
                    lines.append(f"  {str(val)[:50]}: {cnt}")
            except TypeError:
                lines.append("Contains nested objects — showing samples:")
                for v in series.dropna().head(5):
                    lines.append(f"  {str(v)[:100]}")
        return "\n".join(lines), None, None

    # ── run_pandas_query ──────────────────────────────────────
    elif tool_name == "run_pandas_query":
        expression = tool_input["expression"]
        if any(kw in expression for kw in BLOCKED_KEYWORDS):
            return "Blocked: expression contains disallowed operations.", None, None
        try:
            result = eval(expression, {"df": df, "pd": pd})
            if isinstance(result, pd.DataFrame):
                return result.head(20).to_string(), None, None
            elif isinstance(result, pd.Series):
                return result.head(20).to_string(), None, None
            else:
                return str(result), None, None
        except Exception:
            return f"Execution error:\n{traceback.format_exc(limit=3)}", None, None

    # ── create_chart ──────────────────────────────────────────
    elif tool_name == "create_chart":
        chart_type = tool_input["chart_type"]
        x_col = tool_input["x"]
        y_col = tool_input.get("y")
        title = tool_input["title"]
        top_n = tool_input.get("top_n")

        if x_col not in df.columns:
            return f"Error: column '{x_col}' not found.", None, None
        if y_col and y_col != "count" and y_col not in df.columns:
            return f"Error: column '{y_col}' not found.", None, None

        try:
            if chart_type == "bar":
                if y_col == "count" or y_col is None:
                    counts = df[x_col].value_counts().reset_index()
                    counts.columns = [x_col, "count"]
                    if top_n:
                        counts = counts.head(top_n)
                    fig = px.bar(counts, x=x_col, y="count", title=title)
                else:
                    agg = df.dropna(subset=[x_col, y_col]).groupby(x_col)[y_col].mean().reset_index()
                    if top_n:
                        agg = agg.nlargest(top_n, y_col)
                    fig = px.bar(agg, x=x_col, y=y_col, title=title)
            elif chart_type == "histogram":
                fig = px.histogram(df.dropna(subset=[x_col]), x=x_col, title=title)
            elif chart_type == "scatter":
                if not y_col or y_col == "count":
                    return "scatter requires a numeric y column.", None, None
                fig = px.scatter(df.dropna(subset=[x_col, y_col]), x=x_col, y=y_col, title=title)
            elif chart_type == "box":
                if not y_col:
                    return "box chart requires a y column.", None, None
                fig = px.box(df.dropna(subset=[x_col, y_col]), x=x_col, y=y_col, title=title)
            elif chart_type == "pie":
                counts = df[x_col].value_counts().reset_index()
                counts.columns = [x_col, "count"]
                if top_n:
                    counts = counts.head(top_n)
                fig = px.pie(counts, names=x_col, values="count", title=title)
            else:
                return f"Unknown chart type: {chart_type}", None, None

            fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                font_family="Inter, sans-serif",
                title_font_size=14,
                margin=dict(t=45, b=35, l=35, r=15),
            )
            return f"Chart '{title}' created successfully.", fig, None

        except Exception:
            return f"Chart error:\n{traceback.format_exc(limit=3)}", None, None

    # ── filter_dataframe ──────────────────────────────────────
    elif tool_name == "filter_dataframe":
        expression = tool_input["expression"]
        description = tool_input["description"]
        if any(kw in expression for kw in BLOCKED_KEYWORDS):
            return "Blocked: expression contains disallowed operations.", None, None
        try:
            result = eval(expression, {"df": df, "pd": pd})
            if not isinstance(result, pd.DataFrame):
                return "Expression must return a DataFrame.", None, None
            rows_removed = len(df) - len(result)
            new_df = result.reset_index(drop=True)
            return (
                f"Filter applied: {description}\n"
                f"Rows before: {len(df):,} → after: {len(new_df):,} ({rows_removed:,} removed)",
                None, new_df,
            )
        except Exception:
            return f"Filter error:\n{traceback.format_exc(limit=3)}", None, None

    # ── summarize_findings ────────────────────────────────────
    elif tool_name == "summarize_findings":
        return tool_input["summary"], None, None

    else:
        return f"Unknown tool: {tool_name}", None, None


# ── Agent loop ────────────────────────────────────────────────

def run_agent(
    user_goal: str,
    df: pd.DataFrame,
    openrouter_api_key: str,
    model: str = "anthropic/claude-haiku-4-5",
    max_steps: int = 10,
) -> list[dict]:
    """
    Run the Agent loop using OpenRouter (OpenAI-compatible format).

    Returns a list of "events" for the UI to render:
      {"type": "text",      "content": str}
      {"type": "tool",      "name": str, "input": dict}
      {"type": "result",    "content": str}
      {"type": "figure",    "fig": Figure, "title": str}
      {"type": "df_update", "df": DataFrame, "message": str}
      {"type": "summary",   "content": str}
      {"type": "error",     "content": str}
      {"type": "limit",     "content": str}
      {"type": "_final_df", "df": DataFrame}   ← internal, not rendered
    """
    client = OpenAI(
        api_key=openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
    )

    # Build system prompt with dataset context
    col_summary = ", ".join(df.columns[:30].tolist())
    if len(df.columns) > 30:
        col_summary += f" ... (+{len(df.columns)-30} more)"

    system = (
        "You are an autonomous data analyst agent. You have access to a dataset and a set of tools.\n\n"
        f"Dataset overview:\n"
        f"- Shape: {len(df):,} rows × {len(df.columns)} columns\n"
        f"- Columns: {col_summary}\n\n"
        "Your job:\n"
        "1. Use tools to investigate the data step by step\n"
        "2. Always start with inspect_dataframe to understand the structure\n"
        "3. Use get_column_stats and run_pandas_query to dig into specific questions\n"
        "4. Use create_chart when a visual would help communicate a pattern\n"
        "5. Use filter_dataframe ONLY when the user explicitly asks to modify the data\n"
        "6. Always finish with summarize_findings to give a clear conclusion\n\n"
        "Be methodical. Each tool call should build on the previous result.\n"
        "Always respond in English."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_goal},
    ]

    events: list[dict] = []
    current_df = df.copy()

    for step in range(max_steps):
        # ── Call the API ──────────────────────────────────────
        try:
            response = client.chat.completions.create(
                model=model,
                max_tokens=2048,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",   # let the model decide when to use tools
            )
        except Exception as e:
            events.append({"type": "error", "content": f"API error at step {step+1}: {e}"})
            break

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # ── Capture any text the model wrote ──────────────────
        if message.content and message.content.strip():
            events.append({"type": "text", "content": message.content})

        # ── Process tool calls ────────────────────────────────
        if message.tool_calls:
            # Append the assistant's full message (with tool_calls) to history
            messages.append(message)

            tool_results_for_api = []

            for tc in message.tool_calls:
                tool_name = tc.function.name
                try:
                    tool_input = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    tool_input = {}

                # Record the tool call as a visible event
                events.append({
                    "type": "tool",
                    "name": tool_name,
                    "input": tool_input,
                })

                # Execute the tool
                text_result, fig, updated_df = execute_tool(tool_name, tool_input, current_df)

                # Handle chart
                if fig is not None:
                    events.append({
                        "type": "figure",
                        "fig": fig,
                        "title": tool_input.get("title", "Chart"),
                    })

                # Handle dataset update
                if updated_df is not None:
                    current_df = updated_df
                    events.append({
                        "type": "df_update",
                        "df": updated_df,
                        "message": text_result,
                    })

                # Handle summary (signals the agent is done)
                if tool_name == "summarize_findings":
                    events.append({"type": "summary", "content": text_result})

                events.append({"type": "result", "content": text_result})

                # Collect result to send back to the model
                tool_results_for_api.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": text_result,
                })

            # Add all tool results back into message history
            messages.extend(tool_results_for_api)

        # ── Check if the model is done ────────────────────────
        if finish_reason == "stop" and not message.tool_calls:
            break

    else:
        events.append({
            "type": "limit",
            "content": f"Agent reached the maximum of {max_steps} steps and stopped automatically.",
        })

    # Attach final df for the UI to pick up
    events.append({"type": "_final_df", "df": current_df})
    return events
