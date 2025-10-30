import logging
from fastapi import APIRouter

from base import GenerateDataRequest


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/data-generation-engine")


@router.post("/api/generate")
async def generate_data(
    request: GenerateDataRequest,
):
    pass


