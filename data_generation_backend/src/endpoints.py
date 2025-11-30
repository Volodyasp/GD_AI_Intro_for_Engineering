import logging
from typing import Optional, List, Any, Dict

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Body, Request, Depends
from pydantic import BaseModel

from generation_engine.base import GenerateDataOutput, GenerateDataInput
from generation_engine.pipeline import (
    run_generate_data_flow,
    run_apply_change_flow,
    run_chat_agent_flow
)
from generation_engine.models import NLQRequest, AgentResponse
from generation_engine.guardrails import GuardrailsManager

from dependencies import SessionManagerDep, DBManagerDep
from utils.file_processing import read_data_schema_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/data-generation-engine")


# --- Pydantic Models for JSON Payloads ---

class ApplyChangeRequest(BaseModel):
    table_name: str
    user_prompt: str
    temperature: Optional[float] = 0.2
    current_rows: Optional[List[Dict[str, Any]]] = None

class SaveDataRequest(BaseModel):
    table_name: str
    data: List[Dict[str, Any]]

# --- Dependency ---
def get_guardrails_manager() -> GuardrailsManager:
    return GuardrailsManager()

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
    try:
        ddl_content = await read_data_schema_file(file_schema)
        if not ddl_content and "CREATE TABLE" in user_prompt.upper():
            pass

        flow_input = GenerateDataInput(
            user_prompt=user_prompt,
            ddl_schema=ddl_content,
            llm_config={"temperature": temperature}
        )

        response: GenerateDataOutput = await run_generate_data_flow(flow_input, db_manager)

        if not response.is_success:
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
    table_name: Optional[str] = Form(None),
    user_prompt: Optional[str] = Form(None),
    payload: Optional[ApplyChangeRequest] = Body(None)
):
    try:
        target_table = table_name or (payload.table_name if payload else None)
        target_prompt = user_prompt or (payload.user_prompt if payload else None)

        if not target_table or not target_prompt:
            raise HTTPException(status_code=400, detail="Both 'table_name' and 'user_prompt' are required.")

        logger.info(f"Applying change to table '{target_table}' with prompt: {target_prompt}")

        fetch_query = f"SELECT * FROM {target_table} LIMIT 50"
        current_rows = await db_manager.fetch_data(fetch_query)

        if not current_rows:
            current_rows = []

        new_rows = await run_apply_change_flow(current_rows, target_prompt)

        return {"generated_text": {target_table: new_rows}}

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Apply Change Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/save_data")
async def save_data(
    db_manager: DBManagerDep,
    payload: SaveDataRequest
):
    """
    Saves (Replaces) data for a specific table.
    """
    try:
        logger.info(f"Saving {len(payload.data)} rows to {payload.table_name}")
        success = await db_manager.replace_data(payload.table_name, payload.data)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to save data to {payload.table_name}")
        return {"status": "success", "message": "Data saved successfully"}
    except Exception as e:
        logger.error(f"Save Data Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/api/talk-to-data", response_model=AgentResponse)
async def talk_to_data(
        db_manager: DBManagerDep,
        payload: NLQRequest,
        guardrails: GuardrailsManager = Depends(get_guardrails_manager)
):
    try:
        logger.info(f"Received Agent Request: {payload.user_prompt}")
        result = await run_chat_agent_flow(
            user_prompt=payload.user_prompt,
            chat_history=payload.chat_history,
            db_manager=db_manager,
            guardrails=guardrails
        )
        return result
    except Exception as e:
        logger.error(f"Talk to Data Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
