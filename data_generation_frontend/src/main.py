import json
import re
import time
from typing import Dict, List

import pandas as pd
import streamlit as st

# ---------- Page config ----------
st.set_page_config(
    page_title="Data Assistant",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Helpers ----------
def parse_ddl_tables(ddl_text: str) -> List[str]:
    """
    Naive DDL table name extraction.
    In your real app, replace with sqlglot/pglast or information_schema introspection.
    """
    if not ddl_text:
        return []
    pattern = re.compile(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_\"\.]+)", re.IGNORECASE)
    tables = pattern.findall(ddl_text)
    # Strip optional schema prefix and quotes
    clean = []
    for t in tables:
        t = t.split(".")[-1]
        t = t.strip('"')
        clean.append(t)
    return list(dict.fromkeys(clean))  # de-dupe preserving order

def make_sample_df(name: str) -> pd.DataFrame:
    # You can replace with data preview pulled from your generator or DB.
    return pd.DataFrame(
        [
            {"ID": "001", "Name": "Sample Data 1", "Category": "Category A", "Value": 245.50},
            {"ID": "002", "Name": "Sample Data 2", "Category": "Category B", "Value": 127.80},
            {"ID": "003", "Name": "Sample Data 3", "Category": "Category A", "Value": 389.20},
        ]
    )

def ensure_session_state():
    if "dataframes" not in st.session_state:
        st.session_state.dataframes = {}  # table_name -> pd.DataFrame
    if "tables" not in st.session_state:
        st.session_state.tables = []      # list of table names
    if "last_sql" not in st.session_state:
        st.session_state.last_sql = ""
    if "generation_params" not in st.session_state:
        st.session_state.generation_params = {"temperature": 0.4, "max_tokens": 100}

ensure_session_state()

# ---------- Sidebar Navigation ----------
st.sidebar.title("Data Assistant")
nav = st.sidebar.radio(
    " ",
    options=["Data Generation", "Talk to your data"],
    index=0,
    format_func=lambda s: "💽 " + s if s == "Data Generation" else "💬 " + s,
)

# ---------- Data Generation Page ----------
if nav == "Data Generation":
    st.markdown("### Prompt")
    user_prompt = st.text_input("Enter your prompt here…", key="dg_prompt", label_visibility="collapsed")

    colu1, colu2 = st.columns([1, 2])
    with colu1:
        ddl_file = st.file_uploader("Upload DDL Schema", type=["sql", "ddl", "txt", "json"])
        st.caption("Supported formats: SQL, JSON, TXT")
    with colu2:
        st.text_area("…or paste DDL here", key="ddl_text", height=140)

    st.markdown("### Advanced Parameters")
    colp1, colp2 = st.columns([3, 1])
    with colp1:
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=st.session_state.generation_params["temperature"], step=0.1)
    with colp2:
        max_tokens = st.number_input("Max Tokens", min_value=1, max_value=8192, value=st.session_state.generation_params["max_tokens"], step=10)

    # Generate button
    if st.button("Generate", type="primary"):
        # Store params
        st.session_state.generation_params["temperature"] = float(temperature)
        st.session_state.generation_params["max_tokens"] = int(max_tokens)

        # Resolve DDL text: file takes precedence, then textarea
        ddl_text = ""
        if ddl_file is not None:
            try:
                ddl_text = ddl_file.read().decode("utf-8")
            except Exception:
                ddl_text = ddl_file.read().decode(errors="ignore")
        if not ddl_text:
            ddl_text = st.session_state.get("ddl_text", "")

        # Fallback: if JSON was provided, treat as schema dict (ignored here, but you can map to tables)
        if ddl_file is not None and ddl_file.name.lower().endswith(".json"):
            try:
                _ = json.loads(ddl_text)
            except json.JSONDecodeError:
                st.warning("Uploaded JSON is not valid; continuing with raw text.")

        # Parse tables
        tables = parse_ddl_tables(ddl_text)
        if not tables:
            # Provide a default if no tables were parsed (for demo)
            tables = ["users"]

        # Simulate generation
        with st.spinner("Generating synthetic data…"):
            time.sleep(0.6)
            st.session_state.tables = tables
            st.session_state.dataframes = {t: make_sample_df(t) for t in tables}

        st.success(f"Generated data for {len(st.session_state.tables)} table(s).")

    # ---------- Data Preview ----------
    if st.session_state.tables:
        header_left, header_right = st.columns([3, 1])
        with header_left:
            st.markdown("### Data Preview")
        with header_right:
            selected_table = st.selectbox(" ", options=st.session_state.tables, label_visibility="collapsed", key="preview_table")

        df = st.session_state.dataframes.get(st.session_state.preview_table, pd.DataFrame())
        st.dataframe(df, use_container_width=True)

        # Quick edit form
        with st.form(key="quick_edit_form", clear_on_submit=False):
            st.text_input("Enter quick edit instructions…", key="edit_instr", label_visibility="collapsed")
            submit = st.form_submit_button("Submit")
            if submit:
                instr = st.session_state.get("edit_instr", "").strip()
                if not instr:
                    st.info("Please enter an instruction.")
                else:
                    # Here you would call your LLM/tooling to apply edits.
                    # For demo: pretend we updated the table.
                    st.toast(f"Applied edit to '{st.session_state.preview_table}': {instr}", icon="✅")

# ---------- Talk to your data Page ----------
else:
    st.markdown("### Ask a question about your data")
    q = st.text_input("Ask in natural language (read-only SQL will be generated)", key="nlq", label_visibility="collapsed")
    colb1, colb2 = st.columns([1, 6])
    run_q = colb1.button("Run", type="primary")

    if run_q and not st.session_state.tables:
        st.warning("Generate or load data first on the Data Generation tab.")
    elif run_q:
        # In your app: call Gemini function to make SQL; here, a placeholder
        sql = f"SELECT * FROM {st.session_state.tables[0]} LIMIT 3;"
        st.session_state.last_sql = sql

    if st.session_state.last_sql:
        st.code(st.session_state.last_sql, language="sql")
        # Show a small result table (sample from the selected table or first table)
        table_for_results = st.session_state.get("preview_table") or (st.session_state.tables[0] if st.session_state.tables else "users")
        df = st.session_state.dataframes.get(table_for_results, make_sample_df(table_for_results)).head(3)
        st.dataframe(df, use_container_width=True)
