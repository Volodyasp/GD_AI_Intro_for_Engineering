import base64
import io
import json
import logging
from typing import List

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from config import CONFIG
from database.manager import DBManager
from generation_engine.common import clean_json_string
from generation_engine.embeddings import EmbeddingService
from generation_engine.guardrails import GuardrailsManager
from generation_engine.models import AgentAction, AgentResponse, ChatMessage
from generation_engine.prompts import (
    AGENT_ROUTER_SYSTEM_INSTRUCTION,
    VIZ_CODE_GEN_SYSTEM_INSTRUCTION,
    VIZ_CODE_GEN_USER_PARAMS,
)
from utils.observability import trace_step
from vertexai.generative_models import GenerationConfig, GenerativeModel

logger = logging.getLogger(__name__)


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
            raise ValueError(
                "Visualization code did not produce a 'buf' BytesIO object."
            )
    except Exception as e:
        logger.error(f"Viz Execution failed: {e}")
        plt.clf()
        return ""


@trace_step(name="chat_agent_flow", as_type="generation")
async def run_chat_agent_flow(
    user_prompt: str,
    chat_history: List[ChatMessage],
    db_manager: DBManager,
    guardrails: GuardrailsManager,
) -> AgentResponse:
    # 1. Guardrails Check
    guard_result = await guardrails.check_input(user_prompt)
    if not guard_result.is_safe:
        return AgentResponse(
            type="error", text="I cannot process this request due to safety guidelines."
        )
    if not guard_result.is_relevant:
        return AgentResponse(
            type="text",
            text="I can only help with data analysis, SQL, and visualization questions.",
        )

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
                if row["table_name"] != curr_table:
                    schema_context += f"\nTable: {row['table_name']}\n"
                    curr_table = row["table_name"]
                schema_context += f" - {row['column_name']} ({row['data_type']})\n"
    except Exception:
        schema_context = "Schema unavailable."

    examples_context = ""
    try:
        # Initialize service
        embedding_service = EmbeddingService()

        # Get embedding for user question
        user_vector = await embedding_service.get_embedding(user_prompt)

        if user_vector:
            # Search DB for similar SQL examples
            similar_examples = await db_manager.search_similar_examples(
                user_vector, limit=3
            )

            if similar_examples:
                examples_context = "### RELEVANT SQL EXAMPLES ###\nUse these to understand how to join tables correctly:\n"
                for ex in similar_examples:
                    examples_context += (
                        f"Q: {ex['question']}\nSQL: {ex['sql_query']}\n---\n"
                    )
                logger.info(f"Retrieved {len(similar_examples)} few-shot examples.")
    except Exception as e:
        logger.error(f"Vector search failed (continuing without examples): {e}")

    # 3. Router Decision
    model_name = CONFIG["vertex_ai"]["models"]["generate_data"].get(
        "model_name", "gemini-2.0-flash-exp"
    )

    # We update the system instruction to acknowledge we might provide examples
    llm_router = GenerativeModel(
        model_name=model_name,
        system_instruction=AGENT_ROUTER_SYSTEM_INSTRUCTION.format(
            ddl_schema=schema_context
        ),
    )

    # Construct history context string
    history_str = "\n".join([f"{m.role}: {m.content}" for m in chat_history[-5:]])

    # INJECT THE EXAMPLES INTO THE PROMPT
    router_prompt = f"""
    Session History:
    {history_str}

    {examples_context}

    Current User Request: {user_prompt}
    """

    try:
        logger.info("Routing user request...")
        router_resp = await llm_router.generate_content_async(
            router_prompt,
            generation_config=GenerationConfig(response_mime_type="application/json"),
        )
        action_json = json.loads(
            clean_json_string(router_resp.candidates[0].content.text)
        )
        action = AgentAction(**action_json)
        logger.info(f"Agent decided to: {action.action}")

    except Exception as e:
        logger.error(f"Router failed: {e}")
        return AgentResponse(type="error", text="Sorry, I didn't understand that.")

    # 4. Handle Actions (Logic remains exactly the same as your original code)
    if action.action == "chat":
        return AgentResponse(
            type="text", text=action.response_text or "I'm not sure how to respond."
        )

    elif action.action == "sql":
        if not action.sql_query:
            return AgentResponse(type="error", text="Failed to generate SQL.")

        try:
            results = await db_manager.fetch_data(action.sql_query)
            return AgentResponse(
                type="sql", text=action.thought, sql=action.sql_query, data=results
            )
        except Exception as e:
            return AgentResponse(
                type="error",
                text=f"SQL Execution Error: {str(e)}",
                sql=action.sql_query,
            )

    elif action.action == "visualization":
        if not action.sql_query:
            return AgentResponse(
                type="error", text="No SQL generated for visualization."
            )

        try:
            # A. Get Data
            results = await db_manager.fetch_data(action.sql_query)
            if not results:
                return AgentResponse(
                    type="text", text="The query returned no data to visualize."
                )

            df = pd.DataFrame(results)

            # B. Generate Viz Code
            llm_viz = GenerativeModel(
                model_name=model_name,
                system_instruction=VIZ_CODE_GEN_SYSTEM_INSTRUCTION,
            )
            viz_prompt = VIZ_CODE_GEN_USER_PARAMS.format(
                columns=list(df.columns),
                sample_data=df.head(3).to_dict(),
                viz_description=action.viz_description,
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
                    data=results,
                )
            else:
                return AgentResponse(
                    type="error", text="Failed to generate image from code."
                )

        except Exception as e:
            logger.error(f"Viz flow failed: {e}", exc_info=True)
            return AgentResponse(type="error", text=f"Visualization failed: {str(e)}")

    return AgentResponse(type="error", text="Unknown action.")
