from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class GenerateDataInput(BaseModel):
    user_prompt: str
    ddl_schema: str
    llm_config: Optional[Dict[str, Any]] = None

class GenerateDataOutput(BaseModel):
    generated_text: Dict[str, List[Dict[str, Any]]]
    is_success: bool = True
    error_message: Optional[str] = None
