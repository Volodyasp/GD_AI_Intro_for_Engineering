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
SAVE_DATA_ENDPOINT = f"{BACKEND_URL}/data-generation-engine/api/save_data"
TALK_ENDPOINT = f"{BACKEND_URL}/data-generation-engine/api/talk-to-data"

LOADER_STYLE = os.getenv("LOADER_STYLE", "bike")

st.set_page_config(
    page_title="Data Assistant",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- Helpers ----------
def parse_ddl_tables(ddl_text: str) -> List[str]:
    if not ddl_text: return []
    pattern = re.compile(r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_\"\.]+)", re.IGNORECASE)
    tables = pattern.findall(ddl_text)
    clean = []
    for t in tables:
        t = t.split(".")[-1]
        t = t.strip('"')
        clean.append(t)
    return list(dict.fromkeys(clean))


def convert_dfs_to_zip(dataframes: Dict[str, pd.DataFrame]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for table_name, df in dataframes.items():
            csv_data = df.to_csv(index=False).encode("utf-8")
            zip_file.writestr(f"{table_name}.csv", csv_data)
    return buffer.getvalue()


def ensure_session_state():
    ss = st.session_state
    ss.setdefault("dataframes", {})
    ss.setdefault("tables", [])
    ss.setdefault("generation_params", {"temperature": 0.2, "max_tokens": 8192})
    ss.setdefault("last_request", {})
    ss.setdefault("last_response", None)
    ss.setdefault("apply_prompt", "")
    ss.setdefault("preview_change_df", None)
    ss.setdefault("preview_change_table", None)

    # Chat specific state
    ss.setdefault("messages", [{"role": "assistant",
                                "content": "Hello! I am your Data Assistant. Load some data, then ask me questions about it, or ask me to visualize it."}])


ensure_session_state()


# ---------- Payload Parsing Logic ----------
def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _extract_json_object(s: str) -> str:
    s = _strip_code_fences(s)
    if s.startswith("{") and s.rstrip().endswith("}"): return s
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start: return s[start: end + 1]
    return s


def _load_payload_to_dataset(payload: Any, selected_table: Optional[str] = None) -> Optional[Dict[str, List[dict]]]:
    data = payload
    if isinstance(data, dict) and "generated_text" in data:
        data = data["generated_text"]
    if isinstance(data, str):
        try:
            data = json.loads(_extract_json_object(data))
        except json.JSONDecodeError:
            return None
    if isinstance(data, dict):
        mapped: Dict[str, List[dict]] = {}
        for k, v in data.items():
            if isinstance(v, list):
                mapped[k] = v
        return mapped if mapped else None
    if isinstance(data, list) and selected_table:
        return {selected_table: data}
    return None


def build_preview_from_backend(resp_dict: dict, selected_table: Optional[str] = None) -> bool:
    dataset = _load_payload_to_dataset(resp_dict, selected_table=selected_table)
    if not dataset: return False

    if len(dataset.keys()) > 1 or (not selected_table and len(dataset) == 1):
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
            st.session_state.preview_table = tables[0] if tables else None
        return True
    elif selected_table and selected_table in dataset:
        rows = dataset[selected_table]
        try:
            df = pd.DataFrame(rows)
        except Exception:
            df = pd.json_normalize(rows)
        st.session_state.preview_change_df = df
        st.session_state.preview_change_table = selected_table
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


# ---------- Chat Rendering Helper ----------
def display_chat_message(message):
    with st.chat_message(message["role"]):
        # 1. Display Text
        if message.get("content"):
            st.markdown(message["content"])

        # 2. Display SQL (if agent returned it)
        if message.get("sql"):
            with st.expander("Show SQL Query"):
                st.code(message["sql"], language="sql")

        # 3. Display Data (if agent returned rows)
        if message.get("data"):
            # It might be a list of dicts, convert to DF
            try:
                df_res = pd.DataFrame(message["data"])
                st.dataframe(df_res, use_container_width=True)
            except Exception:
                st.write(message["data"])

        # 4. Display Image (if agent returned base64 image)
        if message.get("image"):
            # Decode base64
            try:
                image_data = base64.b64decode(message["image"])
                st.image(image_data, caption="Generated Visualization")
            except Exception as e:
                st.error(f"Failed to load image: {e}")


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
            min_value=0.0, max_value=1.0,
            value=st.session_state.generation_params["temperature"],
            step=0.1
        )

    # --- ACTION: GENERATE ---
    if st.button("Generate", type="primary", use_container_width=True):
        st.session_state.generation_params["temperature"] = float(temperature)
        st.session_state.preview_change_df = None
        st.session_state.preview_change_table = None

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

                    built = build_preview_from_backend(st.session_state.last_response)
                    if built:
                        status.update(label="Done", state="complete", expanded=False)
                        st.toast("Generation completed", icon="🚴")
                        st.rerun()
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

    # --- DATA PREVIEW SECTION ---
    if st.session_state.tables:
        st.markdown("---")
        header_left, header_right = st.columns([3, 1])
        with header_left:
            st.markdown("### Data Preview")
        with header_right:
            selected_table = st.selectbox("Select Table", options=st.session_state.tables, label_visibility="collapsed",
                                          key="preview_table")

        if st.session_state.preview_change_df is not None and st.session_state.preview_change_table == selected_table:
            st.warning(f"⚠️ You have unsaved changes for table: {selected_table}")
            st.info("Review the modified data below. Click 'Save' to commit to database or 'Revert' to discard.")
            st.dataframe(st.session_state.preview_change_df, use_container_width=True)

            c1, c2 = st.columns(2)
            if c1.button("✅ Save Changes", type="primary", use_container_width=True):
                with st.spinner("Saving changes to database..."):
                    try:
                        save_data = st.session_state.preview_change_df.to_dict(orient="records")
                        save_data_json = json.loads(
                            st.session_state.preview_change_df.to_json(orient="records", date_format="iso"))
                        payload = {"table_name": selected_table, "data": save_data_json}
                        resp = requests.post(SAVE_DATA_ENDPOINT, json=payload, timeout=60)
                        resp.raise_for_status()
                        st.session_state.dataframes[selected_table] = st.session_state.preview_change_df
                        st.session_state.preview_change_df = None
                        st.session_state.preview_change_table = None
                        st.toast("Changes saved successfully!", icon="💾")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save data: {e}")

            if c2.button("❌ Revert", use_container_width=True):
                st.session_state.preview_change_df = None
                st.session_state.preview_change_table = None
                st.toast("Changes discarded.", icon="↩️")
                st.rerun()

        else:
            df = st.session_state.dataframes.get(st.session_state.preview_table, pd.DataFrame())
            st.dataframe(df, use_container_width=True)

            if st.session_state.dataframes:
                zip_bytes = convert_dfs_to_zip(st.session_state.dataframes)
                st.download_button(label="⬇️ Download All Tables (ZIP)", data=zip_bytes, file_name="generated_data.zip",
                                   mime="application/zip")

            st.markdown(f"#### Edit Table: *{st.session_state.preview_table}*")
            with st.form(key="apply_change_form", clear_on_submit=False):
                st.text_area("Describe how to modify this table...", key="apply_prompt", height=80)
                submitted = st.form_submit_button("Preview Changes")
                if submitted:
                    prompt = (st.session_state.get("apply_prompt") or "").strip()
                    if not prompt:
                        st.info("Please enter a change prompt.")
                    else:
                        with st.status(f"Generating preview for '{st.session_state.preview_table}'…", state="running",
                                       expanded=True) as status:
                            render_activity_loader(LOADER_STYLE)
                            try:
                                payload = {"table_name": st.session_state.preview_table, "user_prompt": prompt}
                                resp = requests.post(APPLY_CHANGE_ENDPOINT, data=payload, timeout=90)
                                resp.raise_for_status()
                                ok = build_preview_from_backend(resp.json(),
                                                                selected_table=st.session_state.preview_table)
                                if not ok:
                                    st.error("Could not parse response.")
                                else:
                                    status.update(label="Preview generated", state="complete", expanded=False)
                                    st.rerun()
                            except Exception as e:
                                status.update(label="Request failed", state="error")
                                st.error(f"Error: {e}")

# ---------------------------------------------------------------------
# TAB 2: TALK TO YOUR DATA
# ---------------------------------------------------------------------
else:
    st.markdown("### 💬 Chat with your Data")

    if not st.session_state.tables:
        st.warning(
            "⚠️ No data generated yet! Please go to the 'Data Generation' tab first to create a database schema and data.")

    # Display Chat History
    for msg in st.session_state.messages:
        display_chat_message(msg)

    # Chat Input
    if prompt := st.chat_input("Ask a question about your data..."):
        # 1. Add User Message
        user_msg = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_msg)
        display_chat_message(user_msg)

        # 2. Call Backend
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Prepare history for backend (only role/content needed)
                    history_payload = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages if m["role"] != "system"
                    ]

                    payload = {
                        "user_prompt": prompt,
                        "chat_history": history_payload
                    }

                    response = requests.post(TALK_ENDPOINT, json=payload, timeout=60)
                    response.raise_for_status()
                    agent_res = response.json()

                    # 3. Parse & Add Assistant Message
                    # The backend returns: type, text, sql, data, image
                    assistant_msg = {
                        "role": "assistant",
                        "content": agent_res.get("text", ""),
                        "sql": agent_res.get("sql"),
                        "data": agent_res.get("data"),
                        "image": agent_res.get("image")
                    }

                    # If type is error, maybe prepend an emoji
                    if agent_res.get("type") == "error":
                        assistant_msg["content"] = f"⚠️ {assistant_msg['content']}"

                    st.session_state.messages.append(assistant_msg)

                    # Force a rerun to render the new message properly at the bottom
                    st.rerun()

                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to backend.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
