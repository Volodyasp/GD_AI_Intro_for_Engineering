from fastapi import Depends, Request
from sqlalchemy.sql.annotation import Annotated

from sessions.manager import RedisSessionManager
from database.manager import DBManager


def get_session_manager(request: Request) -> RedisSessionManager:
    return request.app.state.session_manager


def get_db_manager(request: Request):
    return request.app.state.db_manager


SessionManagerDep = Annotated[RedisSessionManager, Depends(get_session_manager)]
DBManagerDep = Annotated[DBManager, Depends(get_db_manager)]
