import logging
import json
from vertexai.generative_models import GenerativeModel, GenerationConfig
from config import CONFIG
from .prompts import GUARDRAILS_SYSTEM_INSTRUCTION
from .models import GuardrailsResult
from utils.observability import trace_step

logger = logging.getLogger(__name__)


class GuardrailsManager:
    def __init__(self):
        # We use a lightweight model (Gemini Flash) for guardrails to keep latency low
        # Fallback to a default if config is missing specific guardrail model
        self.model_name = CONFIG["vertex_ai"]["models"]["generate_data"].get("model_name", "gemini-2.0-flash")

        self.model = GenerativeModel(
            model_name=self.model_name,
            system_instruction=GUARDRAILS_SYSTEM_INSTRUCTION
        )

        # Enforce JSON output matching our Pydantic model
        self.config = GenerationConfig(
            temperature=0.0,  # Deterministic behavior for safety
            response_mime_type="application/json",
            response_schema=GuardrailsResult.model_json_schema()
        )

    @trace_step(name="guardrails_check")
    async def check_input(self, user_prompt: str) -> GuardrailsResult:
        """
        Analyzes user input for safety and relevance.
        """
        try:
            logger.info(f"Running guardrails check on: {user_prompt[:50]}...")

            response = await self.model.generate_content_async(
                user_prompt,
                generation_config=self.config
            )

            # Parse the JSON response strictly using Pydantic
            result_json = json.loads(response.candidates[0].content.text)
            guard_result = GuardrailsResult(**result_json)

            logger.info(f"Guardrails Result: Safe={guard_result.is_safe}, Relevant={guard_result.is_relevant}")
            return guard_result

        except Exception as e:
            logger.error(f"Guardrails check failed: {e}")
            # Fail-safe: If check fails, we proceed with caution (or block, depending on policy).
            return GuardrailsResult(
                is_safe=True,
                is_relevant=True,
                reason="Guardrail check failed system-side, proceeding."
            )
