import logging
from typing import Any, Dict

from config import CONFIG
from database.manager import DBManager
from generation_engine.common import clean_json_string
from generation_engine.prompts import (
    DATA_QUERY_SYSTEM_INSTRUCTION,
    DATA_QUERY_USER_PARAMS,
)
from vertexai.generative_models import GenerativeModel

logger = logging.getLogger(__name__)


async def run_natural_language_query_flow(
    user_prompt: str, db_manager: DBManager
) -> Dict[str, Any]:
    # ... (Keep existing implementation)
    schema_query = """
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """
    try:
        schema_rows = await db_manager.fetch_data(schema_query)
        if not schema_rows:
            schema_context = "No tables found in the database."
        else:
            schema_context = "Active Database Schema:\n"
            curr_table = ""
            for row in schema_rows:
                if row["table_name"] != curr_table:
                    schema_context += f"\nTable: {row['table_name']}\n"
                    curr_table = row["table_name"]
                schema_context += f" - {row['column_name']} ({row['data_type']})\n"
    except Exception as e:
        logger.warning(f"Could not fetch schema context: {e}")
        schema_context = "Schema information unavailable."

    model_name = CONFIG["vertex_ai"]["models"]["generate_data"].get(
        "model_name", "gemini-2.0-flash-exp"
    )
    llm = GenerativeModel(
        model_name=model_name, system_instruction=DATA_QUERY_SYSTEM_INSTRUCTION
    )
    full_prompt = DATA_QUERY_USER_PARAMS.format(
        ddl_schema=schema_context, user_prompt=user_prompt
    )
    try:
        response = await llm.generate_content_async(full_prompt)
        sql_query = clean_json_string(response.candidates[0].content.text)
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        logger.info(f"Generated SQL: {sql_query}")
        results = await db_manager.fetch_data(sql_query)
        return {"sql": sql_query, "results": results}
    except Exception as e:
        logger.error(f"NLQ Flow Failed: {e}")
        raise e
