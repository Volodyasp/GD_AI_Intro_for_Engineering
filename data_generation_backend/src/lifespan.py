import logging
from fastapi import FastAPI

from sessions.manager import RedisSessionManager
from database.manager import DBManager

logger = logging.getLogger(__name__)


def get_session_manager():
    logger.info('Initializing Redis session manager')
    session_manager = RedisSessionManager()
    session_manager.initialize_pool()
    session_manager.get_pool()
    logger.info("Redis session manager and pool connection initialized.")
    return session_manager


def get_db_manager():
    logger.info('Initializing Database manager')
    db_manager = DBManager()
    logger.info("Database manager and pool connection initialized.")
    return db_manager


def get_lifespan(app: FastAPI):
    app.state.session_manager = get_session_manager()
    app.state.db_manager = get_db_manager()
    yield
    pass
