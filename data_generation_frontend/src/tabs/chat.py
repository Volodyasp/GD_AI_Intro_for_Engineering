import requests
import streamlit as st
from src.config import TALK_ENDPOINT
from src.ui.components import display_chat_message


def render_chat_tab():
    st.markdown("### 💬 Chat with your Data")

    if not st.session_state.tables:
        st.warning(
            "⚠️ No data generated yet! Please go to the 'Data Generation' tab first to create a database schema and data."
        )

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
                        for m in st.session_state.messages
                        if m["role"] != "system"
                    ]

                    payload = {"user_prompt": prompt, "chat_history": history_payload}

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
                        "image": agent_res.get("image"),
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
