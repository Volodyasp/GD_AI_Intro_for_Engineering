from typing import List, Dict

from vertexai.generative_models import GenerativeModel

from .base import GenerateDataBase
from utils.file_processing import read_file


async def run_generate_data_flow(request: GenerateDataBase):
    schema_file: str = read_file(request.file_schema)

    return None