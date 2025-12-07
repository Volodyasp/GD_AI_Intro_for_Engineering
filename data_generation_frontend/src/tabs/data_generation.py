import json

import pandas as pd
import requests
import streamlit as st
from src.config import APPLY_CHANGE_ENDPOINT, GENERATE_ENDPOINT, SAVE_DATA_ENDPOINT
from src.utils import build_preview_from_backend, convert_dfs_to_zip


def render_data_generation_tab():
    st.markdown("### Prompt")
    user_prompt = st.text_input(
        "Enter your prompt here…", key="dg_prompt", label_visibility="collapsed"
    )

    colu1, colu2 = st.columns([1, 2])
    with colu1:
        ddl_file = st.file_uploader(
            "Upload DDL Schema", type=["sql", "ddl", "txt", "json"]
        )
        st.caption("Supported formats: SQL, JSON, TXT")
    with colu2:
        ddl_text_area = st.text_area("…or paste DDL here", key="ddl_text", height=140)

    st.markdown("### Advanced Parameters")
    colp1, colp2 = st.columns([3, 1])
    with colp1:
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.generation_params["temperature"],
            step=0.1,
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
            ctype = (
                "application/json" if fname.lower().endswith(".json") else "text/plain"
            )
            files = {"file_schema": (fname, raw, ctype)}
        elif ddl_text_area.strip():
            raw = ddl_text_area.encode("utf-8")
            files = {"file_schema": ("pasted_schema.sql", raw, "text/plain")}

        if not (user_prompt or files):
            st.error("Please provide a prompt or a schema to generate data.")
        else:
            st.session_state.last_request = {
                "url": GENERATE_ENDPOINT,
                "data": data,
                "has_file": files is not None,
            }

            with st.status(
                "Generating data…", state="running", expanded=True
            ) as status:
                # REPLACED: No more animated gif loader
                try:
                    resp = requests.post(
                        GENERATE_ENDPOINT, data=data, files=files, timeout=120
                    )
                    resp.raise_for_status()
                    st.session_state.last_response = resp.json()

                    built = build_preview_from_backend(st.session_state.last_response)
                    if built:
                        status.update(label="Done", state="complete", expanded=False)
                        st.toast("Generation completed", icon="🚴")
                        st.rerun()
                    else:
                        error_msg = st.session_state.last_response.get(
                            "error_message", "Unknown format"
                        )
                        status.update(label="Generation Failed", state="error")
                        st.error(f"Failed to parse backend response: {error_msg}")

                except requests.HTTPError as e:
                    status.update(
                        label=f"Backend error {e.response.status_code}", state="error"
                    )
                    st.error(
                        f"Backend returned {e.response.status_code}: {e.response.text[:600]}"
                    )
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
            selected_table = st.selectbox(
                "Select Table",
                options=st.session_state.tables,
                label_visibility="collapsed",
                key="preview_table",
            )

        if (
            st.session_state.preview_change_df is not None
            and st.session_state.preview_change_table == selected_table
        ):
            st.warning(f"⚠️ You have unsaved changes for table: {selected_table}")
            st.info(
                "Review the modified data below. Click 'Save' to commit to database or 'Revert' to discard."
            )
            st.dataframe(st.session_state.preview_change_df, use_container_width=True)

            c1, c2 = st.columns(2)
            if c1.button("✅ Save Changes", type="primary", use_container_width=True):
                with st.spinner("Saving changes to database..."):
                    try:
                        save_data = st.session_state.preview_change_df.to_dict(
                            orient="records"
                        )
                        save_data_json = json.loads(
                            st.session_state.preview_change_df.to_json(
                                orient="records", date_format="iso"
                            )
                        )
                        payload = {"table_name": selected_table, "data": save_data_json}
                        resp = requests.post(
                            SAVE_DATA_ENDPOINT, json=payload, timeout=60
                        )
                        resp.raise_for_status()
                        st.session_state.dataframes[selected_table] = (
                            st.session_state.preview_change_df
                        )
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
            df = st.session_state.dataframes.get(
                st.session_state.preview_table, pd.DataFrame()
            )
            st.dataframe(df, use_container_width=True)

            if st.session_state.dataframes:
                zip_bytes = convert_dfs_to_zip(st.session_state.dataframes)
                st.download_button(
                    label="⬇️ Download All Tables (ZIP)",
                    data=zip_bytes,
                    file_name="generated_data.zip",
                    mime="application/zip",
                )

            st.markdown(f"#### Edit Table: *{st.session_state.preview_table}*")
            with st.form(key="apply_change_form", clear_on_submit=False):
                st.text_area(
                    "Describe how to modify this table...",
                    key="apply_prompt",
                    height=80,
                )
                submitted = st.form_submit_button("Preview Changes")
                if submitted:
                    prompt = (st.session_state.get("apply_prompt") or "").strip()
                    if not prompt:
                        st.info("Please enter a change prompt.")
                    else:
                        with st.status(
                            f"Generating preview for '{st.session_state.preview_table}'…",
                            state="running",
                            expanded=True,
                        ) as status:
                            try:
                                payload = {
                                    "table_name": st.session_state.preview_table,
                                    "user_prompt": prompt,
                                }
                                resp = requests.post(
                                    APPLY_CHANGE_ENDPOINT, data=payload, timeout=90
                                )
                                resp.raise_for_status()
                                ok = build_preview_from_backend(
                                    resp.json(),
                                    selected_table=st.session_state.preview_table,
                                )
                                if not ok:
                                    st.error("Could not parse response.")
                                else:
                                    status.update(
                                        label="Preview generated",
                                        state="complete",
                                        expanded=False,
                                    )
                                    st.rerun()
                            except Exception as e:
                                status.update(label="Request failed", state="error")
                                st.error(f"Error: {e}")
