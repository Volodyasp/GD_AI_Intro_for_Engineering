import logging
from typing import Optional, List, Any, Dict

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Body, Request
from pydantic import BaseModel

from generation_engine.base import GenerateDataOutput, GenerateDataInput
from generation_engine.pipeline import (
    run_generate_data_flow,
    run_apply_change_flow,
    run_natural_language_query_flow
)
from dependencies import SessionManagerDep, DBManagerDep
from utils.file_processing import read_data_schema_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-generation-engine")


# --- Pydantic Models for JSON Payloads ---

class ApplyChangeRequest(BaseModel):
    table_name: str
    user_prompt: str
    temperature: Optional[float] = 0.2
    # Optional: Pass current rows if the frontend holds the state
    current_rows: Optional[List[Dict[str, Any]]] = None


class NLQRequest(BaseModel):
    user_prompt: str


# --- Endpoints ---

@router.post("/api/generate_data")
async def generate_data(
        session_manager: SessionManagerDep,
        db_manager: DBManagerDep,
        user_prompt: str = Form(...),
        temperature: float = Form(0.0),
        file_schema: Optional[UploadFile] = File(default=None),
):
    """
    Main entry point: Accepts DDL file + User Prompt -> Generates Data -> Inserts to DB.
    """
    try:
        # 1. Read Schema File
        ddl_content = await read_data_schema_file(file_schema)

        # Fallback: if no file, check if user pasted DDL in the prompt text
        # (Basic heuristic: checks for CREATE TABLE)
        if not ddl_content and "CREATE TABLE" in user_prompt.upper():
            logger.info("No file uploaded, but DDL found in user prompt.")
            pass

        # 2. Build Input Object
        flow_input = GenerateDataInput(
            user_prompt=user_prompt,
            ddl_schema=ddl_content,
            # UPDATED: Use llm_config
            llm_config={"temperature": temperature}
        )

        # 3. Run Pipeline
        response: GenerateDataOutput = await run_generate_data_flow(flow_input, db_manager)

        if not response.is_success:
            # We return 500 but strictly speaking it could be 400 (Bad Request/Schema)
            # Passing the error message allows frontend to display it.
            raise HTTPException(status_code=500, detail=response.error_message)

        return response

    except HTTPException as e:
        logger.error(f"HTTP Exception during generate_data: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Unhandled exception during generate_data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/data-apply-change")
async def data_apply_change(
    request: Request,
    db_manager: DBManagerDep,
    # Support Form Data (Streamlit default)
    table_name: Optional[str] = Form(None),
    user_prompt: Optional[str] = Form(None),
    # Support JSON Body (Standard API usage)
    payload: Optional[ApplyChangeRequest] = Body(None)
):
    """
    Modifies existing data based on user instructions.
    """
    try:
        # Resolve inputs from either Form or JSON Body
        target_table = table_name or (payload.table_name if payload else None)
        target_prompt = user_prompt or (payload.user_prompt if payload else None)

        if not target_table or not target_prompt:
            raise HTTPException(status_code=400, detail="Both 'table_name' and 'user_prompt' are required.")

        logger.info(f"Applying change to table '{target_table}' with prompt: {target_prompt}")

        # 1. Fetch Current Data Context
        fetch_query = f"SELECT * FROM {target_table} LIMIT 50"
        current_rows = await db_manager.fetch_data(fetch_query)

        if not current_rows:
            current_rows = []

        # 2. Run LLM Transformation
        new_rows = await run_apply_change_flow(current_rows, target_prompt)

        # 3. Return Preview
        return {"generated_text": {target_table: new_rows}}

    # ADDED: Catch HTTPException separately so 400s aren't converted to 500s
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Apply Change Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/talk-to-data")
async def talk_to_data(
        db_manager: DBManagerDep,
        payload: NLQRequest
):
    """
    Natural Language to SQL.
    User Question -> SQL -> Results
    """
    try:
        logger.info(f"Received NLQ Request: {payload.user_prompt}")

        result = await run_natural_language_query_flow(payload.user_prompt, db_manager)

        # Result contains {"sql": "...", "results": [...]}
        return result

    except Exception as e:
        logger.error(f"Talk to Data Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
