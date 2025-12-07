import asyncio
import logging
from contextlib import asynccontextmanager

import redis
from database.manager import DBManager
from fastapi import FastAPI
from generation_engine.embeddings import EmbeddingService
from generation_engine.few_shot_data import FEW_SHOT_EXAMPLES
from sessions.manager import RedisSessionManager
from utils.observability import check_langfuse_health

logger = logging.getLogger(__name__)

# Background task reference
_readiness_task = None


async def periodic_readiness_check(app: FastAPI, interval: int = 30):
    """Background task that checks service readiness every `interval` seconds."""
    while True:
        await asyncio.sleep(interval)
        try:
            status = {"services": {}}

            # Check Database
            try:
                await app.state.db_manager.fetch_data("SELECT 1")
                status["services"]["database"] = "healthy"
            except Exception as e:
                status["services"]["database"] = f"unhealthy: {e}"

            # Check Redis (using connection pool)
            try:
                pool = app.state.session_manager.get_pool()
                r = redis.Redis(connection_pool=pool)
                r.ping()
                status["services"]["redis"] = "healthy"
            except Exception as e:
                status["services"]["redis"] = f"unhealthy: {e}"

            # Check Langfuse
            langfuse_health = check_langfuse_health()
            status["services"]["langfuse"] = langfuse_health["status"]

            logger.info(f"Readiness check: {status}")
        except Exception as e:
            logger.error(f"Readiness check failed: {e}")


def get_session_manager():
    logger.info("Initializing Redis session manager")
    session_manager = RedisSessionManager()
    session_manager.initialize_pool()
    session_manager.get_pool()
    return session_manager


def get_db_manager():
    logger.info("Initializing Database manager")
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
        existing_rows = await db_manager.fetch_data(
            "SELECT count(*) as cnt FROM sql_examples"
        )
        count = existing_rows[0]["cnt"]

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
                        embedding=vector,
                    )
            logger.info(f"Successfully seeded {len(FEW_SHOT_EXAMPLES)} examples.")
        else:
            logger.info(f"Vector DB already contains {count} examples. Skipping seed.")

    except Exception as e:
        logger.error(f"Failed during Vector DB seeding: {e}")

    # 4. Start periodic readiness check
    global _readiness_task
    _readiness_task = asyncio.create_task(periodic_readiness_check(app, interval=30))
    logger.info("Started periodic readiness check (every 30s)")

    yield

    # Cleanup
    if _readiness_task:
        _readiness_task.cancel()
        try:
            await _readiness_task
        except asyncio.CancelledError:
            pass
        logger.info("Stopped periodic readiness check")

    await app.state.db_manager.engine.dispose()
