import os
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)


class DBManager:
    def __init__(self):
        # Construct connection string from Docker env vars
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        host = os.getenv("POSTGRES_HOST", "postgres")  # 'postgres' is the service name in docker-compose
        port = os.getenv("POSTGRES_PORT", "5432")
        db_name = os.getenv("POSTGRES_DB", "postgres")

        self.database_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"

        # Create Async Engine
        self.engine = create_async_engine(
            self.database_url,
            echo=False,  # Set to True if you want to see raw SQL logs
            future=True
        )

        # Session Factory
        self.async_session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession
        )

    async def check_connection(self) -> bool:
        """Simple health check."""
        try:
            async with self.async_session_factory() as session:
                await session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    async def execute_ddl(self, ddl_script: str) -> bool:
        """
        Executes a raw DDL script (CREATE TABLE, etc.).
        Splits by ';' to handle multiple statements if necessary.
        """
        async with self.async_session_factory() as session:
            try:
                # We wrap the execution in a transaction
                async with session.begin():
                    # Simple split by semicolon to handle multiple CREATE TABLE statements
                    # Note: This is a basic split; complex SQL with semicolons in strings might need a stronger parser
                    statements = [s.strip() for s in ddl_script.split(';') if s.strip()]

                    for stmt in statements:
                        logger.info(f"Executing DDL: {stmt[:50]}...")
                        await session.execute(text(stmt))

                logger.info("DDL executed successfully.")
                return True
            except SQLAlchemyError as e:
                logger.error(f"DDL Execution failed: {e}")
                # Transaction is automatically rolled back by the context manager on error
                raise e

    async def insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> bool:
        """
        Dynamically inserts a list of dictionaries into the specified table.
        """
        if not data:
            logger.warning(f"No data provided for table {table_name}")
            return False

        async with self.async_session_factory() as session:
            try:
                # We assume all dicts in the list have the same keys
                keys = data[0].keys()
                columns = ", ".join(keys)
                placeholders = ", ".join([f":{key}" for key in keys])

                # Construct raw SQL insert
                sql = text(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})")

                async with session.begin():
                    # SQLAlchemy handles list of dicts as bulk insert automatically
                    await session.execute(sql, data)

                logger.info(f"Inserted {len(data)} rows into {table_name}")
                return True
            except SQLAlchemyError as e:
                logger.error(f"Insert failed for {table_name}: {e}")
                raise e

    async def fetch_data(self, query: str) -> List[Dict[str, Any]]:
        """
        Executes a SELECT query and returns list of dictionaries.
        """
        async with self.async_session_factory() as session:
            try:
                result = await session.execute(text(query))
                # Convert rows to dicts
                rows = result.mappings().all()
                return [dict(row) for row in rows]
            except SQLAlchemyError as e:
                logger.error(f"Query execution failed: {e}")
                raise e
