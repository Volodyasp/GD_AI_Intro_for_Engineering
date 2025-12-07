import io
import json
import re
import zipfile
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


def parse_ddl_tables(ddl_text: str) -> List[str]:
    """Extracts table names from DDL text."""
    if not ddl_text:
        return []
    pattern = re.compile(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([a-zA-Z0-9_\"\.]+)", re.IGNORECASE
    )
    tables = pattern.findall(ddl_text)
    clean = []
    for t in tables:
        t = t.split(".")[-1]
        t = t.strip('"')
        clean.append(t)
    return list(dict.fromkeys(clean))


def convert_dfs_to_zip(dataframes: Dict[str, pd.DataFrame]) -> bytes:
    """Zips a dictionary of dataframes into a byte stream."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for table_name, df in dataframes.items():
            csv_data = df.to_csv(index=False).encode("utf-8")
            zip_file.writestr(f"{table_name}.csv", csv_data)
    return buffer.getvalue()


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


def _load_payload_to_dataset(
    payload: Any, selected_table: Optional[str] = None
) -> Optional[Dict[str, List[dict]]]:
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


def build_preview_from_backend(
    resp_dict: dict, selected_table: Optional[str] = None
) -> bool:
    """Parsed backend response and updates Session State with new data."""
    dataset = _load_payload_to_dataset(resp_dict, selected_table=selected_table)
    if not dataset:
        return False

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
        if (
            "preview_table" not in st.session_state
            or st.session_state.preview_table not in tables
        ):
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
