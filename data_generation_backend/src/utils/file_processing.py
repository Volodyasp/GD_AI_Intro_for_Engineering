from typing import Optional
from fastapi import UploadFile


async def read_data_schema_file(file: Optional[UploadFile]) -> str:
    if file is None:
        return ""

    raw = await file.read()
    try:
        decoded_text = raw.decode("utf-8")
    except UnicodeDecodeError:
        decoded_text = raw.decode("latin-1", errors="ignore")

    return decoded_text
