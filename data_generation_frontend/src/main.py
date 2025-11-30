import io
import base64
import json
import os
import re
import zipfile
from typing import Dict, List, Optional, Any

import pandas as pd
import requests
import streamlit as st

# ---------- Config ----------
BACKEND_URL = os.getenv("BACKEND_URL", "http://data-generation-backend:8600")
GENERATE_ENDPOINT = f"{BACKEND_URL}/data-generation-engine/api/generate_data"
APPLY_CHANGE_ENDPOINT = f"{BACKEND_URL}/data-generation-engine/api/data-apply-change"
TALK_ENDPOINT = f"{BACKEND_URL}/data-generation-engine/api/talk-to-data"

LOADER_STYLE = os.getenv("LOADER_STYLE", "bike")  # "bike" | "runner" | path/URL/data-URI to .gif

st.set_page_config(
    page_title="Data Assistant",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Helpers ----------

def parse_ddl_tables(ddl_text: str) -> List[str]:
    """Extracts table names from DDL text using regex."""
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


def convert_dfs_to_zip(dataframes: Dict[str, pd.DataFrame]) -> bytes:
    """Converts a dictionary of DataFrames to a ZIP file containing CSVs."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for table_name, df in dataframes.items():
            csv_data = df.to_csv(index=False).encode("utf-8")
            zip_file.writestr(f"{table_name}.csv", csv_data)
    return buffer.getvalue()


def ensure_session_state():
    """Initializes Streamlit session state variables."""
    ss = st.session_state
    ss.setdefault("dataframes", {})
    ss.setdefault("tables", [])
    ss.setdefault("last_sql", "")
    ss.setdefault("last_nlq_results", [])
    ss.setdefault("generation_params", {"temperature": 0.2, "max_tokens": 8192})
    ss.setdefault("last_request", {})
    ss.setdefault("last_response", None)
    ss.setdefault("show_backend_raw", False)
    ss.setdefault("apply_prompt", "")


ensure_session_state()


# ---------- Payload Parsing Logic ----------

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
        return s[start: end + 1]
    return s


def _load_payload_to_dataset(payload: Any, selected_table: Optional[str] = None) -> Optional[Dict[str, List[dict]]]:
    """
    Normalizes backend response into a standard Dict[Table, List[Rows]].
    Backend usually returns: { "generated_text": { "table1": [...], "table2": [...] } }
    """
    data = payload

    # 1. If wrapped in 'generated_text', unwrap it
    if isinstance(data, dict) and "generated_text" in data:
        data = data["generated_text"]

    # 2. If it's a string, try to parse it
    if isinstance(data, str):
        try:
            data = json.loads(_extract_json_object(data))
        except json.JSONDecodeError:
            return None

    # 3. If it's a dict of lists (Standard Format)
    if isinstance(data, dict):
        mapped: Dict[str, List[dict]] = {}
        for k, v in data.items():
            if isinstance(v, list):
                mapped[k] = v
        return mapped if mapped else None

    # 4. If it's a direct list and we know the table
    if isinstance(data, list) and selected_table:
        return {selected_table: data}

    return None


def build_preview_from_backend(resp_dict: dict, selected_table: Optional[str] = None) -> bool:
    """
    Updates session state with data from backend response.
    Returns True if successful.
    """
    dataset = _load_payload_to_dataset(resp_dict, selected_table=selected_table)

    if not dataset:
        return False

    # Logic:
    # - If multiple tables -> Replace entire dataset (Full Generation)
    # - If single table -> Update just that table (Apply Change)

    if len(dataset.keys()) > 1 or (not selected_table and len(dataset) == 1):
        # Replace all
        tables = []
        dfs: Dict[str, pd.DataFrame] = {}
        for name, rows in dataset.items():
            try:
                # Normalize json to handle nested structures if necessary
                dfs[name] = pd.DataFrame(rows)
            except Exception:
                dfs[name] = pd.json_normalize(rows)
            tables.append(name)

        st.session_state.tables = tables
        st.session_state.dataframes = dfs

        # Set default preview table
        if "preview_table" not in st.session_state or st.session_state.preview_table not in tables:
            st.session_state.preview_table = tables[0] if tables else None
        return True

    elif selected_table and selected_table in dataset:
        # Patch single table
        rows = dataset[selected_table]
        try:
            df = pd.DataFrame(rows)
        except Exception:
            df = pd.json_normalize(rows)

        st.session_state.dataframes[selected_table] = df
        return True

    return False


# ---------- Loader UI ----------
def render_activity_loader(style: str = "bike", width: int = 160):
    s = (style or "").strip()
    if s.lower().endswith(".gif") or s.startswith(("http://", "https://", "file://", "data:", "/")):
        try:
            if s.startswith(("http://", "https://", "data:")):
                st.markdown(f'<img src="{s}" width="{width}">', unsafe_allow_html=True)
            else:
                p = s.replace("file://", "")
                with open(p, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                st.markdown(f'<img src="data:image/gif;base64,{b64}" width="{width}">', unsafe_allow_html=True)
        except Exception:
            _render_emoji_loader("bike")
        return
    _render_emoji_loader("runner" if s.lower() == "runner" else "bike")


def _render_emoji_loader(style: str = "bike"):
    emoji = "🚴‍♂️" if style == "bike" else "🏃‍♂️"
    st.markdown(
        f"""
        <style>
        .emoji-loader {{ font-size: 64px; display: inline-block; animation: ride 0.9s ease-in-out infinite alternate; filter: drop-shadow(0 2px 8px rgba(0,0,0,0.35)); }}
        @keyframes ride {{ 0% {{ transform: translateX(0px) rotate(0deg); }} 50% {{ transform: translateX(10px) rotate(2deg); }} 100% {{ transform: translateX(20px) rotate(-2deg); }} }}
        </style>
        <div class="emoji-loader">{emoji}</div>
        """,
        unsafe_allow_html=True,
    )


# ==========================================
# Main UI Layout
# ==========================================

st.sidebar.title("Data Assistant")
nav = st.sidebar.radio(
    " ",
    options=["Data Generation", "Talk to your data"],
    index=0,
    format_func=lambda s: "💽 " + s if s == "Data Generation" else "💬 " + s,
)

# ---------------------------------------------------------------------
# TAB 1: DATA GENERATION
# ---------------------------------------------------------------------
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
            min_value=0.0, max_value=1.0,  # Vertex AI usually 0-1 or 0-2 depending on model
            value=st.session_state.generation_params["temperature"],
            step=0.1
        )

    # --- ACTION: GENERATE ---
    if st.button("Generate", type="primary", use_container_width=True):
        st.session_state.generation_params["temperature"] = float(temperature)

        # Prepare Payload
        data = {"user_prompt": user_prompt or "", "temperature": str(temperature)}
        files = None

        # Handle Schema Input
        if ddl_file is not None:
            raw = ddl_file.getvalue()
            fname = ddl_file.name
            ctype = "application/json" if fname.lower().endswith(".json") else "text/plain"
            files = {"file_schema": (fname, raw, ctype)}
        elif ddl_text_area.strip():
            raw = ddl_text_area.encode("utf-8")
            files = {"file_schema": ("pasted_schema.sql", raw, "text/plain")}

        # We need at least a prompt or a file
        if not (user_prompt or files):
            st.error("Please provide a prompt or a schema to generate data.")
        else:
            st.session_state.last_request = {"url": GENERATE_ENDPOINT, "data": data, "has_file": files is not None}

            with st.status("Generating data…", state="running", expanded=True) as status:
                render_activity_loader(LOADER_STYLE)
                try:
                    resp = requests.post(GENERATE_ENDPOINT, data=data, files=files, timeout=120)
                    resp.raise_for_status()

                    st.session_state.last_response = resp.json()

                    # Parse Response
                    built = build_preview_from_backend(st.session_state.last_response)

                    if built:
                        status.update(label="Done", state="complete", expanded=False)
                        st.toast("Generation completed", icon="🚴")
                    else:
                        error_msg = st.session_state.last_response.get("error_message", "Unknown format")
                        status.update(label="Generation Failed", state="error")
                        st.error(f"Failed to parse backend response: {error_msg}")

                except requests.HTTPError as e:
                    status.update(label=f"Backend error {e.response.status_code}", state="error")
                    st.error(f"Backend returned {e.response.status_code}: {e.response.text[:600]}")
                except Exception as e:
                    status.update(label="Request failed", state="error")
                    st.error(f"Request failed: {e}")

    # --- Debug: Raw JSON ---
    if st.session_state.last_response is not None:
        cols = st.columns([1, 6])


        def _toggle_raw():
            st.session_state.show_backend_raw = not st.session_state.get("show_backend_raw", False)


        with cols[0]:
            st.button(
                "Show JSON" if not st.session_state.show_backend_raw else "Hide JSON",
                on_click=_toggle_raw
            )
        if st.session_state.show_backend_raw:
            st.json(st.session_state.last_response)

    # --- DATA PREVIEW SECTION ---
    if st.session_state.tables:
        st.markdown("---")
        header_left, header_right = st.columns([3, 1])
        with header_left:
            st.markdown("### Data Preview")
        with header_right:
            # Table Selector
            selected_table = st.selectbox(
                "Select Table",
                options=st.session_state.tables,
                label_visibility="collapsed",
                key="preview_table"
            )

        # Show DataFrame
        df = st.session_state.dataframes.get(st.session_state.preview_table, pd.DataFrame())
        st.dataframe(df, use_container_width=True)

        # --- DOWNLOAD BUTTON ---
        if st.session_state.dataframes:
            zip_bytes = convert_dfs_to_zip(st.session_state.dataframes)
            st.download_button(
                label="⬇️ Download All Tables (ZIP)",
                data=zip_bytes,
                file_name="generated_data.zip",
                mime="application/zip",
            )

        # --- APPLY CHANGE SECTION ---
        st.markdown(f"#### Edit Table: *{st.session_state.preview_table}*")
        with st.form(key="apply_change_form", clear_on_submit=False):
            st.text_area("Describe how to modify this table...", key="apply_prompt", height=80)
            submitted = st.form_submit_button("Apply Changes")

            if submitted:
                prompt = (st.session_state.get("apply_prompt") or "").strip()
                if not prompt:
                    st.info("Please enter a change prompt.")
                else:
                    with st.status(f"Applying change to '{st.session_state.preview_table}'…", state="running",
                                   expanded=True) as status:
                        render_activity_loader(LOADER_STYLE)
                        try:
                            # Send as Form Data (or JSON)
                            payload = {
                                "table_name": st.session_state.preview_table,
                                "user_prompt": prompt,
                            }
                            # FIXED: Use 'data=' instead of 'json=' to match backend Form parameters
                            resp = requests.post(APPLY_CHANGE_ENDPOINT, data=payload, timeout=90)

                            resp.raise_for_status()
                            resp_json = resp.json()

                            # Update frontend dataset
                            ok = build_preview_from_backend(resp_json, selected_table=st.session_state.preview_table)
                            if not ok:
                                st.error("The apply-change response could not be parsed into rows.")
                                status.update(label="Apply failed", state="error")
                            else:
                                status.update(label="Change applied", state="complete", expanded=False)
                                st.toast("Table updated", icon="✅")
                                st.rerun()  # Rerun to refresh the dataframe view

                        except requests.HTTPError as e:
                            status.update(label=f"Backend error {e.response.status_code}", state="error")
                            st.error(f"Backend returned {e.response.status_code}: {e.response.text[:600]}")
                        except Exception as e:
                            status.update(label="Request failed", state="error")
                            st.error(f"Request failed: {e}")

# ---------------------------------------------------------------------
# TAB 2: TALK TO YOUR DATA
# ---------------------------------------------------------------------
else:
    st.markdown("### Ask a question about your data")

    q = st.text_input("Ask in natural language (SQL will be generated)", key="nlq", label_visibility="collapsed")
    colb1, colb2 = st.columns([1, 6])
    run_q = colb1.button("Run Query", type="primary")

    if run_q and not st.session_state.tables:
        st.warning("Please generate data first on the 'Data Generation' tab so the database has tables.")

    elif run_q and q:
        with st.status("Thinking & Querying...", state="running") as status:
            try:
                # Payload matching NLQRequest in Backend
                payload = {"user_prompt": q}

                resp = requests.post(TALK_ENDPOINT, json=payload, timeout=60)
                resp.raise_for_status()

                result = resp.json()
                # Expected format: {"sql": "SELECT ...", "results": [{...}, {...}]}

                st.session_state.last_sql = result.get("sql", "-- No SQL returned")
                st.session_state.last_nlq_results = result.get("results", [])

                status.update(label="Query executed", state="complete")

            except requests.HTTPError as e:
                status.update(label="Error", state="error")
                st.error(f"Backend Error: {e.response.text}")
            except Exception as e:
                status.update(label="Error", state="error")
                st.error(f"Failed to query data: {e}")

    # Display Results
    if st.session_state.get("last_sql"):
        st.markdown("#### Generated SQL")
        st.code(st.session_state.last_sql, language="sql")

        st.markdown("#### Result")
        results = st.session_state.get("last_nlq_results", [])
        if results:
            st.dataframe(pd.DataFrame(results), use_container_width=True)
        else:
            st.info("Query returned no results.")
