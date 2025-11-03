import logging
from datetime import datetime
from typing import Optional, List, Any, Dict

from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Body, Request

from generation_engine.base import GenerateDataOutput, GenerateDataInput
from generation_engine.pipeline import run_generate_data_flow
from utils.file_processing import read_data_schema_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data-generation-engine")


@router.post("/api/generate_data")
async def generate_data(
    user_prompt: str = Form(...),
    temperature: float = Form(0.0),
    file_schema: Optional[UploadFile] = File(default=None),
):
    try:
        try:
            ddl_content = await read_data_schema_file(file_schema)
        except TypeError as e:
            raise HTTPException(
                status_code=400, detail=f"Wrong type of the schema file: {e}"
            )

        flow_input: GenerateDataInput = GenerateDataInput(
            user_prompt=user_prompt,
            ddl_schema=ddl_content,
            generation_config=(
                {"temperature": temperature} if temperature is not None else None
            ),
        )
        response: GenerateDataOutput = await run_generate_data_flow(flow_input)
        return response

    except HTTPException as e:
        logger.error(f"HTTPException occurred during generate data: {e}", exc_info=e)
        raise e

    except Exception as e:
        logger.error(f"Unhandled exception during generate data: {e}", exc_info=e)
        raise HTTPException(status_code=500, detail=str(e))





### DUMMY DUMMY DUMMY
from pydantic import BaseModel

class ApplyChangeJSON(BaseModel):
    table_name: Optional[str] = None
    user_prompt: Optional[str] = None
    temperature: Optional[float] = None
    current_table: Optional[List[Dict[str, Any]]] = None
    current_dataset: Optional[Dict[str, List[Dict[str, Any]]]] = None


def _make_dummy_rows(table_name: str, user_prompt: str) -> List[Dict[str, Any]]:
    """Produce a few demo rows that ‘reflect’ the prompt."""
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    return [
        {
            "id": 1,
            "name": f"{table_name} Row A",
            "prompt_applied": user_prompt,
            "updated_at": now,
        },
        {
            "id": 2,
            "name": f"{table_name} Row B",
            "prompt_applied": user_prompt,
            "updated_at": now,
        },
        {
            "id": 3,
            "name": f"{table_name} Row C",
            "prompt_applied": user_prompt,
            "updated_at": now,
        },
    ]


@router.post("/api/data-apply-change")
async def data_apply_change(
    request: Request,
    # Form fields (so your current frontend `data=...` works)
    table_name_form: Optional[str] = Form(None),
    user_prompt_form: Optional[str] = Form(None),
    temperature_form: Optional[float] = Form(None),
    # Optional JSON body (so you can also send JSON if you want)
    json_body: Optional[ApplyChangeJSON] = Body(None),
):
    """
    Dummy endpoint that accepts EITHER form-data OR JSON.
    Returns a payload compatible with the frontend's build_preview_from_backend():
      { "generated_text": { <table_name>: [ {row}, ... ] } }
    """

    # Prefer form fields if present; otherwise use JSON body
    table_name = table_name_form or (json_body.table_name if json_body else None)
    user_prompt = user_prompt_form or (json_body.user_prompt if json_body else None)
    temperature = temperature_form if temperature_form is not None else (
        json_body.temperature if json_body else None
    )

    if not table_name or not user_prompt:
        raise HTTPException(status_code=400, detail="Both 'table_name' and 'user_prompt' are required.")

    # Optional: access current table/dataset if you want to do real edits later
    current_table = json_body.current_table if (json_body and json_body.current_table) else None
    current_dataset = json_body.current_dataset if (json_body and json_body.current_dataset) else None

    logger.info(
        "Apply-change called | table=%s | temp=%s | prompt=%s | has_current_table=%s | has_current_dataset=%s",
        table_name,
        temperature,
        user_prompt[:200],
        current_table is not None,
        current_dataset is not None,
    )

    # ---- DUMMY GENERATION LOGIC ----
    # For now just return 3 synthetic rows for the requested table.
    # Replace this with your LLM / business logic.
    new_rows = _make_dummy_rows(table_name, user_prompt)

    # If you want to sometimes return the WHOLE dataset, build it here:
    #   generated = { "Restaurants": new_rows, "Orders": [...], ... }
    # For single-table patch (what the frontend expects by default), return just one key:
    generated = {table_name: new_rows}

    return {"generated_text": generated}
