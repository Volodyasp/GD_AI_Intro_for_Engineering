import streamlit as st
from src.tabs.chat import render_chat_tab
from src.tabs.data_generation import render_data_generation_tab

# ---------- Config ----------
st.set_page_config(
    page_title="Data Assistant",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)


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
    ss.setdefault("preview_table", None)

    # Chat specific state
    ss.setdefault(
        "messages",
        [
            {
                "role": "assistant",
                "content": "Hello! I am your Data Assistant. Load some data, then ask me questions about it, or ask me to visualize it.",
            }
        ],
    )


ensure_session_state()

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

if nav == "Data Generation":
    render_data_generation_tab()
else:
    render_chat_tab()
