import logging
from typing import List, Dict

from vertexai.generative_models import GenerativeModel, GenerationConfig

from .base import GenerateDataInput, GenerateDataOutput
from .prompts import DATA_GENERATE_SYSTEM_INSTRUCTION, DATA_GENERATE_USER_PARAMS
from config import CONFIG
from utils.file_processing import read_data_schema_file


logger = logging.getLogger(__name__)


async def run_generate_data_flow(request: GenerateDataInput) -> GenerateDataOutput:
    user_prompt: str = request.user_prompt
    ddl_schema: str = await read_data_schema_file(request.ddl_schema)
    user_generation_config = request.model_config if request.model_config else {}

    generation_config: GenerationConfig = GenerationConfig.from_dict({
        **CONFIG["vertex_ai"]["models"]["generate_data"]["generation_config"],
        **user_generation_config
    })
    llm = GenerativeModel(
        model_name=CONFIG["vertex_ai"]["models"]["generate_data"]["model_name"],
        system_instruction=DATA_GENERATE_SYSTEM_INSTRUCTION
    )
    try:
        prompt = DATA_GENERATE_USER_PARAMS.format(user_prompt=user_prompt, ddl_schema=ddl_schema)
        logger.info(f"Full user prompt: {prompt}")
        response = await llm.generate_content_async(prompt, generation_config=generation_config)
        text_output: str = response.candidates[0].content.text
        logger.info(f"Generation response: {text_output}")
    except Exception as e:
        text_output = "Error occurred during generation"
        logger.error(f"Error occurred during llm generation {e}", exc_info=True)

    output = GenerateDataOutput(generated_text=text_output)
    return output
