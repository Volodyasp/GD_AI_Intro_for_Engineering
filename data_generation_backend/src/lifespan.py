import logging
from fastapi import FastAPI
from contextlib import asynccontextmanager

from sessions.manager import RedisSessionManager
from database.manager import DBManager
from generation_engine.embeddings import EmbeddingService
from generation_engine.few_shot_data import FEW_SHOT_EXAMPLES

logger = logging.getLogger(__name__)


def get_session_manager():
    logger.info('Initializing Redis session manager')
    session_manager = RedisSessionManager()
    session_manager.initialize_pool()
    session_manager.get_pool()
    return session_manager


def get_db_manager():
    logger.info('Initializing Database manager')
    db_manager = DBManager()
    return db_manager


@asynccontextmanager
async def get_lifespan(app: FastAPI):
    # 1. Initialize Managers
    app.state.session_manager = get_session_manager()
    db_manager = get_db_manager()
    app.state.db_manager = db_manager

    # 2. Ensure DB Connection & Vector Extension
    await db_manager.check_connection()

    # 3. Seed Vector DB (Step 2 of Phase 3)
    # Check if we already have examples
    try:
        existing_rows = await db_manager.fetch_data("SELECT count(*) as cnt FROM sql_examples")
        count = existing_rows[0]['cnt']

        if count == 0:
            logger.info("Vector DB is empty. Seeding few-shot examples...")
            embedding_service = EmbeddingService()

            for example in FEW_SHOT_EXAMPLES:
                # Generate embedding for the question
                vector = await embedding_service.get_embedding(example["question"])
                if vector:
                    # Insert into Postgres
                    await db_manager.insert_example(
                        question=example["question"],
                        sql_query=example["sql_query"],
                        embedding=vector
                    )
            logger.info(f"Successfully seeded {len(FEW_SHOT_EXAMPLES)} examples.")
        else:
            logger.info(f"Vector DB already contains {count} examples. Skipping seed.")

    except Exception as e:
        logger.error(f"Failed during Vector DB seeding: {e}")

    yield

    await app.state.db_manager.engine.dispose()
