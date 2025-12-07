import base64

import pandas as pd
import streamlit as st


def display_chat_message(message):
    """Renders a single chat message with optional SQL, Data, or Image."""
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
