import logging
from typing import List
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from config import CONFIG

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        # Using the multimodal embedding model or gecko for text
        self.model_name = "text-embedding-004"
        self.model = TextEmbeddingModel.from_pretrained(self.model_name)

    async def get_embedding(self, text: str) -> List[float]:
        """Generates a vector embedding for a single text string."""
        try:
            # Vertex AI expects a list of inputs
            inputs = [TextEmbeddingInput(text, "RETRIEVAL_QUERY")]
            embeddings = await self.model.get_embeddings_async(inputs)
            return embeddings[0].values
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return []

    async def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of strings (for database seeding)."""
        try:
            inputs = [TextEmbeddingInput(t, "RETRIEVAL_DOCUMENT") for t in texts]
            # Vertex API has limits on batch size (usually 250), keeping it simple here
            embeddings = await self.model.get_embeddings_async(inputs)
            return [e.values for e in embeddings]
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return []
