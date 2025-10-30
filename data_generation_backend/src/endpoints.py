import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, Form
from pathlib import Path

#from base import GenerateDataRequest
from generation_engine.pipeline import run_generate_data_flow
from generation_engine.base import GenerateDataOutput, GenerateDataInput


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data-generation-engine")


ALLOWED_EXT = {".sql", ".ddl", ".txt"}


@router.post("/api/generate_data")
async def generate_data(
    user_prompt: str = Form(...),
    temperature: float = Form(0.0),
    file_schema: UploadFile | None = File(default=None), # Made typing more explicit
):
    try:
        ddl_content = None

        if file_schema and file_schema.filename:
            ext = Path(file_schema.filename).suffix.lower()
            if ext not in ALLOWED_EXT:
                raise HTTPException(
                    status_code=400,
                    detail=f"File extension '{ext}' not allowed. Allowed are: {ALLOWED_EXT}"
                )

            ddl_content_bytes = await file_schema.read()
            ddl_content = ddl_content_bytes.decode('utf-8')


        flow_input: GenerateDataInput = GenerateDataInput(
            user_prompt=user_prompt,
            ddl_schema=ddl_content,
            generation_config={"temperature": temperature} if temperature is not None else None,
        )
        response: GenerateDataOutput = run_generate_data_flow(flow_input)
        return response

    except HTTPException as e:
        logger.error(f"HTTPException occurred during generate data: {e}", exc_info=e)
        raise e

    except Exception as e:
        logger.error(f"Unhandled exception during generate data: {e}", exc_info=e)
        raise HTTPException(status_code=500, detail=str(e))
