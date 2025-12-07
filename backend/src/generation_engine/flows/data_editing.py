import json
import logging
from typing import Any, Dict, List

from config import CONFIG
from generation_engine.common import clean_json_string
from generation_engine.prompts import (
    DATA_EDIT_SYSTEM_INSTRUCTION,
    DATA_EDIT_USER_PARAMS,
)
from vertexai.generative_models import GenerationConfig, GenerativeModel

logger = logging.getLogger(__name__)


async def run_apply_change_flow(
    current_data: List[Dict[str, Any]], user_prompt: str
) -> List[Dict[str, Any]]:
    """
    Takes existing rows (JSON) + User Edit Prompt -> Returns Updated Rows (JSON)
    """
    model_name = CONFIG["vertex_ai"]["models"]["generate_data"].get(
        "model_name", "gemini-2.0-flash-exp"
    )
    llm = GenerativeModel(
        model_name=model_name, system_instruction=DATA_EDIT_SYSTEM_INSTRUCTION
    )

    # Lower temperature for editing tasks to ensure adherence to existing schema
    generation_config = GenerationConfig(
        temperature=0.2, response_mime_type="application/json"
    )

    # --- FIX: Add default=str to handle datetime objects from DB ---
    full_prompt = DATA_EDIT_USER_PARAMS.format(
        current_data=json.dumps(current_data, default=str), user_prompt=user_prompt
    )

    try:
        response = await llm.generate_content_async(
            full_prompt, generation_config=generation_config
        )
        cleaned_json = clean_json_string(response.candidates[0].content.text)
        new_data = json.loads(cleaned_json)

        # Ensure we return a list of dicts
        if isinstance(new_data, dict) and len(new_data) == 1:
            # Handle case where LLM wraps it in { "table": [...] }
            return list(new_data.values())[0]

        return new_data
    except Exception as e:
        logger.error(f"Edit Data Failed: {e}")
        raise e
