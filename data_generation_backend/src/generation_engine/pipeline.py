import json
import logging
from typing import Dict, Any, List, Optional

from vertexai.generative_models import GenerativeModel, GenerationConfig
from google.api_core.exceptions import GoogleAPIError

from .base import GenerateDataInput, GenerateDataOutput
from .prompts import (
    DATA_GENERATE_SYSTEM_INSTRUCTION,
    DATA_GENERATE_USER_PARAMS,
    DATA_EDIT_SYSTEM_INSTRUCTION,
    DATA_EDIT_USER_PARAMS,
    DATA_QUERY_SYSTEM_INSTRUCTION,
    DATA_QUERY_USER_PARAMS
)
from config import CONFIG
from database.manager import DBManager

logger = logging.getLogger(__name__)


def clean_json_string(text: str) -> str:
    """
    Cleans the LLM output to ensure it is parseable JSON.
    Removes markdown code fences like ```json and ```.
    """
    text = text.strip()
    # Remove opening fence
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]

    # Remove closing fence
    if text.endswith("```"):
        text = text[:-3]

    return text.strip()


async def run_generate_data_flow(
        request: GenerateDataInput,
        db_manager: DBManager
) -> GenerateDataOutput:
    """
    1. Executes DDL in DB (to create tables).
    2. Generates Synthetic Data via Gemini.
    3. Inserts Data into DB.
    """
    user_prompt: str = request.user_prompt
    ddl_schema: str = request.ddl_schema

    # --- 1. Execute DDL Schema ---
    # We do this FIRST to ensure the schema is valid SQL before spending tokens.
    if ddl_schema:
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

    # --- 2. Configure Gemini ---
    model_name = CONFIG["vertex_ai"]["models"]["generate_data"].get("model_name", "gemini-2.0-flash-exp")

    user_gen_config = request.model_config if request.model_config else {}

    # Force JSON response for structured data generation
    generation_config = GenerationConfig.from_dict({
        **CONFIG["vertex_ai"]["models"]["generate_data"]["generation_config"],
        **user_gen_config,
        "response_mime_type": "application/json"
    })

    llm = GenerativeModel(
        model_name=model_name,
        system_instruction=DATA_GENERATE_SYSTEM_INSTRUCTION
    )

    # --- 3. Generate Content ---
    generated_data: Dict[str, List[Dict[str, Any]]] = {}
    try:
        full_prompt = DATA_GENERATE_USER_PARAMS.format(
            user_prompt=user_prompt,
            ddl_schema=ddl_schema
        )

        logger.info(f"Sending generation request to {model_name}...")
        response = await llm.generate_content_async(full_prompt, generation_config=generation_config)

        raw_text = response.candidates[0].content.text
        cleaned_json = clean_json_string(raw_text)

        # Parse JSON
        generated_data = json.loads(cleaned_json)
        logger.info(f"LLM Generated data for tables: {list(generated_data.keys())}")

    except json.JSONDecodeError as e:
        logger.error(f"LLM produced invalid JSON: {e}. Raw text: {raw_text[:100]}...")
        return GenerateDataOutput(
            generated_text={},
            is_success=False,
            error_message="AI generation failed to produce valid JSON."
        )
    except Exception as e:
        logger.error(f"Unexpected error during generation: {e}", exc_info=True)
        return GenerateDataOutput(
            generated_text={},
            is_success=False,
            error_message=f"System Error: {str(e)}"
        )

    # --- 4. Insert Data into DB ---
    try:
        for table_name, rows in generated_data.items():
            if rows:
                logger.info(f"Inserting {len(rows)} rows into {table_name}...")
                await db_manager.insert_data(table_name, rows)
            else:
                logger.warning(f"Table {table_name} generated but has no rows.")

    except Exception as e:
        logger.error(f"Data Insertion Failed: {e}")
        # We return the generated text even if DB insert fails, so the user can see what happened
        return GenerateDataOutput(
            generated_text=generated_data,
            is_success=False,
            error_message=f"Data generated but failed to save to DB: {str(e)}"
        )

    return GenerateDataOutput(generated_text=generated_data, is_success=True)


async def run_apply_change_flow(
        current_data: List[Dict[str, Any]],
        user_prompt: str
) -> List[Dict[str, Any]]:
    """
    Takes existing rows (JSON) + User Edit Prompt -> Returns Updated Rows (JSON)
    """
    model_name = CONFIG["vertex_ai"]["models"]["generate_data"].get("model_name", "gemini-2.0-flash-exp")
    llm = GenerativeModel(
        model_name=model_name,
        system_instruction=DATA_EDIT_SYSTEM_INSTRUCTION
    )

    # Lower temperature for editing tasks to ensure adherence to existing schema
    generation_config = GenerationConfig(
        temperature=0.2,
        response_mime_type="application/json"
    )

    full_prompt = DATA_EDIT_USER_PARAMS.format(
        current_data=json.dumps(current_data),
        user_prompt=user_prompt
    )

    try:
        response = await llm.generate_content_async(full_prompt, generation_config=generation_config)
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


async def run_natural_language_query_flow(
        user_prompt: str,
        db_manager: DBManager
) -> Dict[str, Any]:
    """
    1. Fetches current DB Schema (Table names/Columns).
    2. Generates SQL based on User Question.
    3. Executes SQL and returns results.
    """

    # 1. Fetch Active Schema Context
    # We query Postgres to see what tables/columns actually exist
    schema_query = """
        SELECT table_name, column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """
    try:
        schema_rows = await db_manager.fetch_data(schema_query)

        # Format schema for LLM Context
        if not schema_rows:
            schema_context = "No tables found in the database."
        else:
            schema_context = "Active Database Schema:\n"
            curr_table = ""
            for row in schema_rows:
                if row['table_name'] != curr_table:
                    schema_context += f"\nTable: {row['table_name']}\n"
                    curr_table = row['table_name']
                schema_context += f" - {row['column_name']} ({row['data_type']})\n"
    except Exception as e:
        logger.warning(f"Could not fetch schema context: {e}")
        schema_context = "Schema information unavailable."

    # 2. Setup Gemini for SQL Generation
    model_name = CONFIG["vertex_ai"]["models"]["generate_data"].get("model_name", "gemini-2.0-flash-exp")
    llm = GenerativeModel(
        model_name=model_name,
        system_instruction=DATA_QUERY_SYSTEM_INSTRUCTION
    )

    full_prompt = DATA_QUERY_USER_PARAMS.format(
        ddl_schema=schema_context,
        user_prompt=user_prompt
    )

    try:
        # Generate SQL
        response = await llm.generate_content_async(full_prompt)
        sql_query = clean_json_string(response.candidates[0].content.text)

        # Extra cleanup for SQL blocks
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

        logger.info(f"Generated SQL: {sql_query}")

        # Execute SQL
        results = await db_manager.fetch_data(sql_query)

        return {
            "sql": sql_query,
            "results": results
        }
    except Exception as e:
        logger.error(f"NLQ Flow Failed: {e}")
        raise e
