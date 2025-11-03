import io
import base64
import json
import os
import re
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

# ---------- Config ----------
BACKEND_URL = os.getenv("BACKEND_URL", "http://data-generation-backend:8600")
GENERATE_ENDPOINT = f"{BACKEND_URL}/data-generation-engine/api/generate_data"
APPLY_CHANGE_ENDPOINT = f"{BACKEND_URL}/data-generation-engine/api/data-apply-change"
LOADER_STYLE = os.getenv("LOADER_STYLE", "bike")  # "bike" | "runner" | path/URL/data-URI to .gif

st.set_page_config(
    page_title="Data Assistant",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Helpers ----------
def parse_ddl_tables(ddl_text: str) -> List[str]:
    if not ddl_text:
        return []
    pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_\"\.]+)",
        re.IGNORECASE,
    )
    tables = pattern.findall(ddl_text)
    clean = []
    for t in tables:
        t = t.split(".")[-1]
        t = t.strip('"')
        clean.append(t)
    return list(dict.fromkeys(clean))

def make_sample_df(name: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"ID": "001", "Name": "Sample Data 1", "Category": "Category A", "Value": 245.50},
            {"ID": "002", "Name": "Sample Data 2", "Category": "Category B", "Value": 127.80},
            {"ID": "003", "Name": "Sample Data 3", "Category": "Category A", "Value": 389.20},
        ]
    )

def ensure_session_state():
    ss = st.session_state
    ss.setdefault("dataframes", {})
    ss.setdefault("tables", [])
    ss.setdefault("last_sql", "")
    ss.setdefault("generation_params", {"temperature": 0.4, "max_tokens": 100})
    ss.setdefault("last_request", {})
    ss.setdefault("last_response", None)
    ss.setdefault("show_backend_raw", False)
    ss.setdefault("apply_prompt", "")

ensure_session_state()

# ---------- Backend payload -> tables & dataframes ----------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()

def _extract_json_object(s: str) -> str:
    s = _strip_code_fences(s)
    if s.startswith("{") and s.rstrip().endswith("}"):
        return s
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s

def _json_relax(s: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", s)

def _load_payload_to_dataset(payload, selected_table: Optional[str] = None) -> Optional[Dict[str, List[dict]]]:
    """
    Accepts:
      - dict of {table: [rows]}
      - list of rows (applies to selected_table)
    Returns:
      dict {table: [rows]} or None on failure.
    """
    data = payload
    if isinstance(data, str):
        try:
            data = json.loads(_extract_json_object(data))
        except json.JSONDecodeError:
            try:
                data = json.loads(_json_relax(_extract_json_object(data)))
            except Exception:
                return None

    if isinstance(data, dict):
        # assume dict-of-lists
        mapped: Dict[str, List[dict]] = {}
        for k, v in data.items():
            if isinstance(v, list):
                mapped[k] = v
        return mapped if mapped else None

    if isinstance(data, list) and selected_table:
        return {selected_table: data}

    return None

def build_preview_from_backend(resp_dict: dict, selected_table: Optional[str] = None) -> bool:
    """
    Parses backend response into tables/dataframes.
    If it contains the whole dataset -> replace all tables.
    If it contains only the selected table -> replace that table.
    """
    payload = resp_dict.get("generated_text")
    if payload is None:
        return False

    dataset = _load_payload_to_dataset(payload, selected_table=selected_table)
    if dataset is None:
        return False

    # Replace entire set if multiple tables; otherwise patch single table.
    if len(dataset.keys()) > 1:
        tables = []
        dfs: Dict[str, pd.DataFrame] = {}
        for name, rows in dataset.items():
            try:
                dfs[name] = pd.DataFrame(rows)
            except Exception:
                dfs[name] = pd.json_normalize(rows)
            tables.append(name)
        st.session_state.tables = tables
        st.session_state.dataframes = dfs
        if "preview_table" not in st.session_state or st.session_state.preview_table not in tables:
            st.session_state.preview_table = tables[0]
        return True
    else:
        (name, rows), = dataset.items()
        try:
            df = pd.DataFrame(rows)
        except Exception:
            df = pd.json_normalize(rows)
        st.session_state.dataframes[name] = df
        if name not in st.session_state.tables:
            st.session_state.tables.append(name)
        if "preview_table" not in st.session_state or st.session_state.preview_table not in st.session_state.tables:
            st.session_state.preview_table = name
        return True

# ---------- Loader (GIF or emoji) ----------
def render_activity_loader(style: str = "bike", width: int = 160):
    s = (style or "").strip()

    # Treat as GIF (path/URL/data-URI)
    if s.lower().endswith(".gif") or s.startswith(("http://", "https://", "file://", "data:", "/")):
        try:
            if s.startswith(("http://", "https://", "data:")):
                st.markdown(f'<img src="{s}" width="{width}">', unsafe_allow_html=True)
            else:
                p = s.replace("file://", "")
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                st.markdown(f'<img src="data:image/gif;base64,{b64}" width="{width}">', unsafe_allow_html=True)
        except Exception as e:
            st.info(f"Loader GIF error: {e}; falling back to emoji.")
            _render_emoji_loader("bike")
        return

    _render_emoji_loader("runner" if s.lower() == "runner" else "bike")

def _render_emoji_loader(style: str = "bike"):
    emoji = "🚴‍♂️" if style == "bike" else "🏃‍♂️"
    st.markdown(
        f"""
        <style>
        .emoji-loader {{
            font-size: 64px;
            display: inline-block;
            animation: ride 0.9s ease-in-out infinite alternate;
            filter: drop-shadow(0 2px 8px rgba(0,0,0,0.35));
        }}
        @keyframes ride {{
            0%   {{ transform: translateX(0px) rotate(0deg);   }}
            50%  {{ transform: translateX(10px) rotate(2deg);  }}
            100% {{ transform: translateX(20px) rotate(-2deg); }}
        }}
        </style>
        <div class="emoji-loader">{emoji}</div>
        """,
        unsafe_allow_html=True,
    )

# ---------- Sidebar ----------
st.sidebar.title("Data Assistant")
nav = st.sidebar.radio(
    " ",
    options=["Data Generation", "Talk to your data"],
    index=0,
    format_func=lambda s: "💽 " + s if s == "Data Generation" else "💬 " + s,
)

# ---------- Data Generation ----------
if nav == "Data Generation":
    st.markdown("### Prompt")
    user_prompt = st.text_input("Enter your prompt here…", key="dg_prompt", label_visibility="collapsed")

    colu1, colu2 = st.columns([1, 2])
    with colu1:
        ddl_file = st.file_uploader("Upload DDL Schema", type=["sql", "ddl", "txt", "json"])
        st.caption("Supported formats: SQL, JSON, TXT")
    with colu2:
        ddl_text_area = st.text_area("…or paste DDL here", key="ddl_text", height=140)

    st.markdown("### Advanced Parameters")
    colp1, colp2 = st.columns([3, 1])
    with colp1:
        temperature = st.slider(
            "Temperature",
            min_value=0.0, max_value=2.0,
            value=st.session_state.generation_params["temperature"],
            step=0.1
        )
    with colp2:
        max_tokens = st.number_input(
            "Max Tokens (frontend only demo)",
            min_value=1, max_value=8192,
            value=st.session_state.generation_params["max_tokens"],
            step=10
        )

    # Generate -> call backend and load dataset
    if st.button("Generate", type="primary", use_container_width=True):
        st.session_state.generation_params["temperature"] = float(temperature)
        st.session_state.generation_params["max_tokens"] = int(max_tokens)

        data = {"user_prompt": user_prompt or "", "temperature": str(temperature)}
        files = None
        if ddl_file is not None:
            raw = ddl_file.getvalue()
            fname = ddl_file.name
            ctype = "application/json" if fname.lower().endswith(".json") else "text/plain"
            files = {"file_schema": (fname, raw, ctype)}
        elif ddl_text_area.strip():
            raw = ddl_text_area.encode("utf-8")
            files = {"file_schema": ("pasted_schema.sql", raw, "text/plain")}

        st.session_state.last_request = {"url": GENERATE_ENDPOINT, "data": data, "has_file": files is not None}

        with st.status("Generating data…", state="running", expanded=True) as status:
            render_activity_loader(LOADER_STYLE)
            try:
                resp = requests.post(GENERATE_ENDPOINT, data=data, files=files, timeout=90)
                resp.raise_for_status()
                st.session_state.last_response = resp.json()
                status.update(label="Done", state="complete", expanded=False)
                st.toast("Generation completed", icon="🚴" if LOADER_STYLE == "bike" else "🏃")
            except requests.HTTPError as e:
                status.update(label=f"Backend error {e.response.status_code}", state="error")
                st.error(f"Backend returned {e.response.status_code}: {e.response.text[:600]}")
                st.stop()
            except Exception as e:
                status.update(label="Request failed", state="error")
                st.error(f"Request failed: {e}")
                st.stop()

        built = build_preview_from_backend(st.session_state.last_response)
        if not built:
            ddl_text_for_preview: Optional[str] = None
            if ddl_file is not None:
                try:
                    ddl_text_for_preview = ddl_file.getvalue().decode("utf-8", errors="ignore")
                except Exception:
                    ddl_text_for_preview = None
            elif ddl_text_area:
                ddl_text_for_preview = ddl_text_area

            tables = parse_ddl_tables(ddl_text_for_preview or "")
            st.session_state.tables = tables or ["preview_table"]
            st.session_state.dataframes = {t: make_sample_df(t) for t in st.session_state.tables}

    # Backend raw JSON toggle (hidden by default)
    if st.session_state.last_response is not None:
        cols = st.columns([1, 6])
        def _toggle_raw():
            st.session_state.show_backend_raw = not st.session_state.get("show_backend_raw", False)
        btn_label = "Show backend JSON" if not st.session_state.show_backend_raw else "Hide backend JSON"
        with cols[0]:
            st.button(btn_label, on_click=_toggle_raw)
        if st.session_state.show_backend_raw:
            st.json(st.session_state.last_response)

    # Preview + Apply change
    if st.session_state.tables:
        header_left, header_right = st.columns([3, 1])
        with header_left:
            st.markdown("### Data Preview")
        with header_right:
            selected_table = st.selectbox(" ", options=st.session_state.tables, label_visibility="collapsed", key="preview_table")

        df = st.session_state.dataframes.get(st.session_state.preview_table, pd.DataFrame())
        st.dataframe(df, use_container_width=True)

        # Apply change form for the selected table
        with st.form(key="apply_change_form", clear_on_submit=False):
            st.text_area("Describe the change to apply to the selected table", key="apply_prompt", height=80)
            submitted = st.form_submit_button("Submit")
            if submitted:
                prompt = (st.session_state.get("apply_prompt") or "").strip()
                if not prompt:
                    st.info("Please enter a change prompt.")
                else:
                    with st.status(f"Applying change to '{st.session_state.preview_table}'…", state="running", expanded=True) as status:
                        render_activity_loader(LOADER_STYLE)
                        try:
                            # Send as simple form data; adapt if your backend expects JSON
                            payload = {
                                "table_name": st.session_state.preview_table,
                                "user_prompt": prompt,
                            }
                            resp = requests.post(APPLY_CHANGE_ENDPOINT, data=payload, timeout=90)
                            resp.raise_for_status()
                            resp_json = resp.json()

                            # Overwrite frontend dataset from the response
                            ok = build_preview_from_backend(resp_json, selected_table=st.session_state.preview_table)
                            if not ok:
                                st.error("The apply-change response could not be parsed into rows.")
                                status.update(label="Apply failed", state="error")
                                st.stop()

                            # Update the “backend raw” preview to reflect the latest response
                            st.session_state.last_response = resp_json

                            status.update(label="Change applied", state="complete", expanded=False)
                            st.toast("Table updated", icon="🚴" if LOADER_STYLE == "bike" else "🏃")

                        except requests.HTTPError as e:
                            status.update(label=f"Backend error {e.response.status_code}", state="error")
                            st.error(f"Backend returned {e.response.status_code}: {e.response.text[:600]}")
                        except Exception as e:
                            status.update(label="Request failed", state="error")
                            st.error(f"Request failed: {e}")

# ---------- Talk to your data ----------
else:
    st.markdown("### Ask a question about your data")
    q = st.text_input("Ask in natural language (read-only SQL will be generated)", key="nlq", label_visibility="collapsed")
    colb1, colb2 = st.columns([1, 6])
    run_q = colb1.button("Run", type="primary")

    if run_q and not st.session_state.tables:
        st.warning("Generate or load data first on the Data Generation tab.")
    elif run_q:
        sql = f"SELECT * FROM {st.session_state.tables[0]} LIMIT 3;"
        st.session_state.last_sql = sql

    if st.session_state.last_sql:
        st.code(st.session_state.last_sql, language="sql")
        table_for_results = st.session_state.get("preview_table") or (st.session_state.tables[0] if st.session_state.tables else "users")
        df = st.session_state.dataframes.get(table_for_results, make_sample_df(table_for_results)).head(3)
        st.dataframe(df, use_container_width=True)
