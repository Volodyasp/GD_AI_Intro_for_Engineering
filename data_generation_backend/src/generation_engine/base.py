from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import UploadFile


class GenerateDataInput(BaseModel):
    user_prompt: str
    ddl_schema: Optional[UploadFile] = None
    generation_config: Optional[Dict] = None


class GenerateDataOutput(BaseModel):
    generated_text: str = ""
