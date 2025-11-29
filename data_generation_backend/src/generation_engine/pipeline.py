import json
import logging
from typing import Dict, Any, List

from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.api_core.exceptions import GoogleAPIError

from .base import GenerateDataInput, GenerateDataOutput
from .prompts import DATA_GENERATE_SYSTEM_INSTRUCTION, DATA_GENERATE_USER_PARAMS
from config import CONFIG
from database.manager import DBManager

logger = logging.getLogger(__name__)


def clean_json_string(text: str) -> str:
    """Removes markdown fencing if Gemini adds it."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


async def run_generate_data_flow(
        request: GenerateDataInput,
        db_manager: DBManager
) -> GenerateDataOutput:
    user_prompt: str = request.user_prompt
    ddl_schema: str = request.ddl_schema

    # 1. Execute DDL to create tables in Postgres
    # We do this FIRST to ensure the schema is valid before wasting tokens on generation.
    try:
        logger.info("Executing DDL Schema in Database...")
        await db_manager.execute_ddl(ddl_schema)
    except Exception as e:
        logger.error(f"DDL Execution Failed: {e}")
        return GenerateDataOutput(
            generated_text={},
            is_success=False,
            error_message=f"Invalid DDL Schema: {str(e)}"
        )

    # 2. Setup Gemini Config
    # We default to the model defined in config, or fallback to a known version
    model_name = CONFIG["vertex_ai"]["models"]["generate_data"].get("model_name", "gemini-2.0-flash-exp")

    user_gen_config = request.model_config if request.model_config else {}

    generation_config = GenerationConfig.from_dict({
        **CONFIG["vertex_ai"]["models"]["generate_data"]["generation_config"],
        **user_gen_config,
        "response_mime_type": "application/json"  # Enforce JSON output
    })

    llm = GenerativeModel(
        model_name=model_name,
        system_instruction=DATA_GENERATE_SYSTEM_INSTRUCTION
    )

    # 3. Call LLM
    try:
        full_prompt = DATA_GENERATE_USER_PARAMS.format(
            user_prompt=user_prompt,
            ddl_schema=ddl_schema
        )

        logger.info(f"Sending prompt to {model_name}...")
        response = await llm.generate_content_async(full_prompt, generation_config=generation_config)

        raw_text = response.candidates[0].content.text
        cleaned_json = clean_json_string(raw_text)

        # Parse JSON to ensure validity
        generated_data: Dict[str, List[Dict[str, Any]]] = json.loads(cleaned_json)
        logger.info(f"LLM Generated data for tables: {list(generated_data.keys())}")

    except json.JSONDecodeError as e:
        logger.error(f"LLM produced invalid JSON: {e}")
        return GenerateDataOutput(
            generated_text={},
            is_success=False,
            error_message="AI generation failed to produce valid JSON."
        )
    except GoogleAPIError as e:
        logger.error(f"Vertex AI Error: {e}")
        return GenerateDataOutput(
            generated_text={},
            is_success=False,
            error_message=f"Vertex AI Error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during generation: {e}", exc_info=True)
        return GenerateDataOutput(
            generated_text={},
            is_success=False,
            error_message=f"System Error: {str(e)}"
        )

    # 4. Insert Generated Data into Postgres
    try:
        for table_name, rows in generated_data.items():
            if rows:
                logger.info(f"Inserting {len(rows)} rows into {table_name}...")
                await db_manager.insert_data(table_name, rows)
            else:
                logger.warning(f"Table {table_name} generated but has no rows.")

    except Exception as e:
        logger.error(f"Data Insertion Failed: {e}")
        return GenerateDataOutput(
            generated_text=generated_data,  # Return what we have, even if DB insert failed
            is_success=False,
            error_message=f"Data generated but failed to save to DB: {str(e)}"
        )

    return GenerateDataOutput(generated_text=generated_data, is_success=True)
