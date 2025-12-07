from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field

# --- Chat Structure ---
class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str

# --- Guardrails Output ---
class GuardrailsResult(BaseModel):
    is_safe: bool = Field(description="True if the prompt contains no malicious content, prompt injection, or jailbreaks.")
    is_relevant: bool = Field(description="True if the prompt is related to data analysis, SQL, visualization, or general greetings.")
    reason: str = Field(description="Short explanation for the classification.")

# --- Router/Agent Output ---
class AgentAction(BaseModel):
    action: Literal["sql", "visualization", "chat"] = Field(description="The action to take.")
    thought: str = Field(description="Internal reasoning for why this action was chosen.")
    sql_query: Optional[str] = Field(None, description="Valid SQL query if action is 'sql'.")
    viz_description: Optional[str] = Field(None, description="Description of the chart if action is 'visualization'.")
    response_text: Optional[str] = Field(None, description="Natural language response to the user.")

# --- API Payloads ---
class NLQRequest(BaseModel):
    user_prompt: str
    chat_history: List[ChatMessage] = []

class AgentResponse(BaseModel):
    type: Literal["text", "sql", "visualization", "error"]
    text: str
    sql: Optional[str] = None
    data: Optional[List[Dict[str, Any]]] = None
    image: Optional[str] = None  # Base64 string
