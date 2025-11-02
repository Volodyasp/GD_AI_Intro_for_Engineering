import logging
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

logger = logging.getLogger(__name__)


ALLOWED_EXT = {".sql", ".ddl", ".txt"}


async def read_data_schema_file(file_schema: Optional[UploadFile]) -> str:
    try:
        ddl_content = ""
        if file_schema and file_schema.filename:
            ext = Path(file_schema.filename).suffix.lower()
            if ext not in ALLOWED_EXT:
                raise TypeError(
                    f"Unsupported file extension: {ext}. Allowed extensions: {ALLOWED_EXT}."
                )

            ddl_content_bytes = await file_schema.read()
            ddl_content = ddl_content_bytes.decode("utf-8")
            return ddl_content

        return ddl_content
    except Exception as e:
        logger.error(f"Error occurred during reading schema file {e}")
        return ""
