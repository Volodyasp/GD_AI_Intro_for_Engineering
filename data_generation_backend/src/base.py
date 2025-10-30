from typing import Optional

from pydantic import BaseModel


class GenerateDataRequest(BaseModel):
    file_schema: Optional[str]
    user_prompt: str
    temperature: float