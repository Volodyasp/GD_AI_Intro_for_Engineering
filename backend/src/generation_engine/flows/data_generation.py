import json
import logging
from typing import Any, Dict, List

from config import CONFIG
from database.manager import DBManager
from generation_engine.base import GenerateDataInput, GenerateDataOutput
from generation_engine.common import clean_json_string
from generation_engine.prompts import (
    DATA_GENERATE_SYSTEM_INSTRUCTION,
    DATA_GENERATE_USER_PARAMS,
    DDL_CONVERSION_SYSTEM_INSTRUCTION,
    DDL_CONVERSION_USER_PARAMS,
)
from vertexai.generative_models import GenerationConfig, GenerativeModel

logger = logging.getLogger(__name__)


async def convert_ddl_to_postgres(ddl_content: str) -> str:
    """
    Uses Gemini to translate incoming DDL (MySQL, Oracle, etc.)
    into standard PostgreSQL DDL.
    """
    model_name = CONFIG["vertex_ai"]["models"]["generate_data"].get(
        "model_name", "gemini-2.0-flash-exp"
    )
    llm = GenerativeModel(
        model_name=model_name, system_instruction=DDL_CONVERSION_SYSTEM_INSTRUCTION
    )

    # Simple config for code generation
    config = GenerationConfig(temperature=0.0)

    prompt = DDL_CONVERSION_USER_PARAMS.format(ddl_schema=ddl_content)

    try:
        logger.info("Normalizing user DDL to PostgreSQL dialect...")
        response = await llm.generate_content_async(prompt, generation_config=config)
        cleaned_ddl = clean_json_string(response.candidates[0].content.text)
        return cleaned_ddl
    except Exception as e:
        logger.error(f"DDL Conversion failed: {e}")
        # Fallback to original if LLM fails (though unlikely to work if dialect differs)
        return ddl_content


async def run_generate_data_flow(
    request: GenerateDataInput, db_manager: DBManager
) -> GenerateDataOutput:
    """
    1. Normalize DDL to Postgres.
    2. Execute DDL.
    3. Generate & Insert Data.
    """
    user_prompt: str = request.user_prompt
    ddl_schema_input: str = request.ddl_schema

    # [STEP 1] Normalize DDL
    if ddl_schema_input:
        final_ddl_schema = await convert_ddl_to_postgres(ddl_schema_input)
        logger.info(f"Final Executed DDL:\n{final_ddl_schema[:200]}...")

        # [STEP 2] Execute DDL
        try:
            logger.info("Executing DDL Schema in Database...")
            await db_manager.execute_ddl(final_ddl_schema)
        except Exception as e:
            logger.error(f"DDL Execution Failed: {e}")
            return GenerateDataOutput(
                generated_text={},
                is_success=False,
                error_message=f"Invalid DDL (even after conversion): {str(e)}",
            )
    else:
        # Should ideally error out if no schema provided
        final_ddl_schema = ""

    # [STEP 3] Configure Gemini for Data Generation
    model_name = CONFIG["vertex_ai"]["models"]["generate_data"].get(
        "model_name", "gemini-2.0-flash-exp"
    )
    user_gen_config = request.llm_config if request.llm_config else {}

    generation_config = GenerationConfig.from_dict(
        {
            **CONFIG["vertex_ai"]["models"]["generate_data"]["generation_config"],
            **user_gen_config,
            "response_mime_type": "application/json",
        }
    )

    llm = GenerativeModel(
        model_name=model_name, system_instruction=DATA_GENERATE_SYSTEM_INSTRUCTION
    )

    # [STEP 4] Generate Content
    generated_data: Dict[str, List[Dict[str, Any]]] = {}
    try:
        # We pass the *original* DDL to the LLM for data generation context
        # (LLMs understand MySQL syntax fine for context),
        # OR pass the converted one. Converted is usually safer for consistency.
        full_prompt = DATA_GENERATE_USER_PARAMS.format(
            user_prompt=user_prompt, ddl_schema=final_ddl_schema
        )

        logger.info(f"Sending generation request to {model_name}...")
        response = await llm.generate_content_async(
            full_prompt, generation_config=generation_config
        )

        raw_text = response.candidates[0].content.text
        cleaned_json = clean_json_string(raw_text)

        generated_data = json.loads(cleaned_json)
        logger.info(f"LLM Generated data for tables: {list(generated_data.keys())}")

    except json.JSONDecodeError as e:
        logger.error(f"LLM produced invalid JSON: {e}")
        return GenerateDataOutput(
            generated_text={},
            is_success=False,
            error_message="AI generation failed to produce valid JSON.",
        )
    except Exception as e:
        logger.error(f"Unexpected error during generation: {e}", exc_info=True)
        return GenerateDataOutput(
            generated_text={}, is_success=False, error_message=f"System Error: {str(e)}"
        )

    # [STEP 5] Insert Data
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
            generated_text=generated_data,
            is_success=False,
            error_message=f"Data generated but failed to save to DB: {str(e)}",
        )

    return GenerateDataOutput(generated_text=generated_data, is_success=True)
