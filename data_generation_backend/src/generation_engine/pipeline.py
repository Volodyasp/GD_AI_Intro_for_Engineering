import json
import logging
import base64
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional

from vertexai.generative_models import GenerativeModel, GenerationConfig
from .base import GenerateDataInput, GenerateDataOutput
from .models import AgentAction, AgentResponse, ChatMessage
from .guardrails import GuardrailsManager
from .prompts import (
    DATA_GENERATE_SYSTEM_INSTRUCTION, DATA_GENERATE_USER_PARAMS,
    DATA_EDIT_SYSTEM_INSTRUCTION, DATA_EDIT_USER_PARAMS,
    DATA_QUERY_SYSTEM_INSTRUCTION, DATA_QUERY_USER_PARAMS,
    DDL_CONVERSION_SYSTEM_INSTRUCTION, DDL_CONVERSION_USER_PARAMS,
    AGENT_ROUTER_SYSTEM_INSTRUCTION,
    VIZ_CODE_GEN_SYSTEM_INSTRUCTION, VIZ_CODE_GEN_USER_PARAMS
)
from config import CONFIG
from database.manager import DBManager
from utils.observability import trace_step

logger = logging.getLogger(__name__)

def clean_json_string(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"): text = text[7:]
    elif text.startswith("```sql"): text = text[6:]
    elif text.startswith("```python"): text = text[9:]
    elif text.startswith("```"): text = text[3:]
    if text.endswith("```"): text = text[:-3]
    return text.strip()


# --- NEW HELPER FUNCTION ---
async def convert_ddl_to_postgres(ddl_content: str) -> str:
    """
    Uses Gemini to translate incoming DDL (MySQL, Oracle, etc.) 
    into standard PostgreSQL DDL.
    """
    model_name = CONFIG["vertex_ai"]["models"]["generate_data"].get("model_name", "gemini-2.0-flash-exp")
    llm = GenerativeModel(
        model_name=model_name,
        system_instruction=DDL_CONVERSION_SYSTEM_INSTRUCTION
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


# --- UPDATED MAIN FLOW ---
async def run_generate_data_flow(
        request: GenerateDataInput,
        db_manager: DBManager
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
                error_message=f"Invalid DDL (even after conversion): {str(e)}"
            )
    else:
        # Should ideally error out if no schema provided
        final_ddl_schema = ""

    # [STEP 3] Configure Gemini for Data Generation
    model_name = CONFIG["vertex_ai"]["models"]["generate_data"].get("model_name", "gemini-2.0-flash-exp")
    user_gen_config = request.llm_config if request.llm_config else {}

    generation_config = GenerationConfig.from_dict({
        **CONFIG["vertex_ai"]["models"]["generate_data"]["generation_config"],
        **user_gen_config,
        "response_mime_type": "application/json"
    })

    llm = GenerativeModel(
        model_name=model_name,
        system_instruction=DATA_GENERATE_SYSTEM_INSTRUCTION
    )

    # [STEP 4] Generate Content
    generated_data: Dict[str, List[Dict[str, Any]]] = {}
    try:
        # We pass the *original* DDL to the LLM for data generation context 
        # (LLMs understand MySQL syntax fine for context), 
        # OR pass the converted one. Converted is usually safer for consistency.
        full_prompt = DATA_GENERATE_USER_PARAMS.format(
            user_prompt=user_prompt,
            ddl_schema=final_ddl_schema
        )

        logger.info(f"Sending generation request to {model_name}...")
        response = await llm.generate_content_async(full_prompt, generation_config=generation_config)

        raw_text = response.candidates[0].content.text
        cleaned_json = clean_json_string(raw_text)

        generated_data = json.loads(cleaned_json)
        logger.info(f"LLM Generated data for tables: {list(generated_data.keys())}")

    except json.JSONDecodeError as e:
        logger.error(f"LLM produced invalid JSON: {e}")
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
            error_message=f"Data generated but failed to save to DB: {str(e)}"
        )

    return GenerateDataOutput(generated_text=generated_data, is_success=True)


# ... (Keep convert_apply_change_flow and run_natural_language_query_flow as is)
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

    # --- FIX: Add default=str to handle datetime objects from DB ---
    full_prompt = DATA_EDIT_USER_PARAMS.format(
        current_data=json.dumps(current_data, default=str),
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
                if row['table_name'] != curr_table:
                    schema_context += f"\nTable: {row['table_name']}\n"
                    curr_table = row['table_name']
                schema_context += f" - {row['column_name']} ({row['data_type']})\n"
    except Exception as e:
        logger.warning(f"Could not fetch schema context: {e}")
        schema_context = "Schema information unavailable."

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
        response = await llm.generate_content_async(full_prompt)
        sql_query = clean_json_string(response.candidates[0].content.text)
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        logger.info(f"Generated SQL: {sql_query}")
        results = await db_manager.fetch_data(sql_query)
        return {
            "sql": sql_query,
            "results": results
        }
    except Exception as e:
        logger.error(f"NLQ Flow Failed: {e}")
        raise e


@trace_step(name="execute_visualization")
def execute_generated_viz_code(code: str, df: pd.DataFrame) -> str:
    """Executes the generated seaborn code and returns base64 image."""
    local_scope = {"df": df, "sns": sns, "plt": plt, "io": io}

    # Safety: execution of arbitrary code is risky. In production, use a sandboxed environment (e.g., E2B).
    # For this practice project, we use `exec` with a restricted scope.
    try:
        exec(code, {}, local_scope)

        if "buf" in local_scope and isinstance(local_scope["buf"], io.BytesIO):
            buf = local_scope["buf"]
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            plt.clf()  # Clear state
            return b64
        else:
            raise ValueError("Visualization code did not produce a 'buf' BytesIO object.")
    except Exception as e:
        logger.error(f"Viz Execution failed: {e}")
        plt.clf()
        return ""


@trace_step(name="chat_agent_flow")
async def run_chat_agent_flow(
        user_prompt: str,
        chat_history: List[ChatMessage],
        db_manager: DBManager,
        guardrails: GuardrailsManager
) -> AgentResponse:
    # 1. Guardrails Check
    guard_result = await guardrails.check_input(user_prompt)
    if not guard_result.is_safe:
        return AgentResponse(type="error", text="I cannot process this request due to safety guidelines.")
    if not guard_result.is_relevant:
        return AgentResponse(type="text", text="I can only help with data analysis, SQL, and visualization questions.")

    # 2. Fetch Schema
    schema_query = """
        SELECT table_name, column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """
    try:
        schema_rows = await db_manager.fetch_data(schema_query)
        if not schema_rows:
            schema_context = "No tables found."
        else:
            schema_context = "Active Database Schema:\n"
            curr_table = ""
            for row in schema_rows:
                if row['table_name'] != curr_table:
                    schema_context += f"\nTable: {row['table_name']}\n"
                    curr_table = row['table_name']
                schema_context += f" - {row['column_name']} ({row['data_type']})\n"
    except Exception:
        schema_context = "Schema unavailable."

    # 3. Router Decision
    model_name = CONFIG["vertex_ai"]["models"]["generate_data"].get("model_name", "gemini-2.0-flash-exp")
    llm_router = GenerativeModel(
        model_name=model_name,
        system_instruction=AGENT_ROUTER_SYSTEM_INSTRUCTION.format(ddl_schema=schema_context)
    )

    # Construct history context string
    history_str = "\n".join([f"{m.role}: {m.content}" for m in chat_history[-5:]])
    router_prompt = f"Session History:\n{history_str}\n\nCurrent User Request: {user_prompt}"

    try:
        logger.info("Routing user request...")
        router_resp = await llm_router.generate_content_async(
            router_prompt,
            generation_config=GenerationConfig(response_mime_type="application/json")
        )
        action_json = json.loads(clean_json_string(router_resp.candidates[0].content.text))
        action = AgentAction(**action_json)
        logger.info(f"Agent decided to: {action.action}")

    except Exception as e:
        logger.error(f"Router failed: {e}")
        return AgentResponse(type="error", text="Sorry, I didn't understand that.")

    # 4. Handle Actions
    if action.action == "chat":
        return AgentResponse(type="text", text=action.response_text or "I'm not sure how to respond.")

    elif action.action == "sql":
        if not action.sql_query:
            return AgentResponse(type="error", text="Failed to generate SQL.")

        try:
            results = await db_manager.fetch_data(action.sql_query)
            # Limit result size for context window
            summary = str(results[:5]) if results else "No results"

            # Generate a summary response using the LLM (Optional, but nice)
            # For now, just returning the data
            return AgentResponse(
                type="sql",
                text=action.thought,
                sql=action.sql_query,
                data=results
            )
        except Exception as e:
            return AgentResponse(type="error", text=f"SQL Execution Error: {str(e)}", sql=action.sql_query)

    elif action.action == "visualization":
        if not action.sql_query:
            return AgentResponse(type="error", text="No SQL generated for visualization.")

        try:
            # A. Get Data
            results = await db_manager.fetch_data(action.sql_query)
            if not results:
                return AgentResponse(type="text", text="The query returned no data to visualize.")

            df = pd.DataFrame(results)

            # B. Generate Viz Code
            llm_viz = GenerativeModel(
                model_name=model_name,
                system_instruction=VIZ_CODE_GEN_SYSTEM_INSTRUCTION
            )
            viz_prompt = VIZ_CODE_GEN_USER_PARAMS.format(
                columns=list(df.columns),
                sample_data=df.head(3).to_dict(),
                viz_description=action.viz_description
            )

            viz_resp = await llm_viz.generate_content_async(viz_prompt)
            code = clean_json_string(viz_resp.candidates[0].content.text)

            # C. Execute Viz Code
            logger.info("Executing visualization code...")
            b64_image = execute_generated_viz_code(code, df)

            if b64_image:
                return AgentResponse(
                    type="visualization",
                    text=f"Generated chart for: {action.viz_description}",
                    sql=action.sql_query,
                    image=b64_image,
                    data=results  # Optional: send raw data too
                )
            else:
                return AgentResponse(type="error", text="Failed to generate image from code.")

        except Exception as e:
            logger.error(f"Viz flow failed: {e}", exc_info=True)
            return AgentResponse(type="error", text=f"Visualization failed: {str(e)}")

    return AgentResponse(type="error", text="Unknown action.")
