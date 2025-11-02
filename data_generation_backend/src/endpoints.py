import logging
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Form

from generation_engine.base import GenerateDataOutput, GenerateDataInput
from generation_engine.pipeline import run_generate_data_flow
from utils.file_processing import read_data_schema_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data-generation-engine")


@router.post("/api/generate_data")
async def generate_data(
    user_prompt: str = Form(...),
    temperature: float = Form(0.0),
    file_schema: Optional[UploadFile] | str = File(default=None),
):
    try:
        try:
            ddl_content = read_data_schema_file(file_schema)
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
