import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date
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
                    statements = [s.strip() for s in ddl_script.split(';') if s.strip()]

                    for stmt in statements:
                        logger.info(f"Executing DDL: {stmt[:50]}...")
                        await session.execute(text(stmt))

                logger.info("DDL executed successfully.")
                return True
            except SQLAlchemyError as e:
                logger.error(f"DDL Execution failed: {e}")
                raise e

    def _convert_value(self, value: Any) -> Any:
        """
        Helper to convert strings into Python objects (datetime, date, bool).
        """
        if isinstance(value, str):
            # Check for Boolean (Common issue with LLM generation)
            if value.upper() == 'TRUE':
                return True
            if value.upper() == 'FALSE':
                return False

            # Check for Datetime
            if "T" in value or (":" in value and " " in value):
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    pass

            # Check for Date
            if len(value) == 10 and value.count("-") == 2:
                try:
                    return date.fromisoformat(value)
                except ValueError:
                    pass
        return value

    async def insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> bool:
        """Dynamically inserts a list of dictionaries into the specified table."""
        if not data:
            logger.warning(f"No data provided for table {table_name}")
            return False

        async with self.async_session_factory() as session:
            try:
                # 1. Normalize data
                all_keys = set().union(*(d.keys() for d in data))
                normalized_data = []
                for row in data:
                    new_row = {}
                    for k in all_keys:
                        raw_value = row.get(k, None)
                        new_row[k] = self._convert_value(raw_value)
                    normalized_data.append(new_row)

                # 2. Prepare SQL
                columns = ", ".join(all_keys)
                placeholders = ", ".join([f":{key}" for key in all_keys])

                sql = text(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})")

                async with session.begin():
                    await session.execute(sql, normalized_data)

                logger.info(f"Inserted {len(data)} rows into {table_name}")
                return True
            except SQLAlchemyError as e:
                logger.error(f"Insert failed for {table_name}: {e}")
                raise e

    async def replace_data(self, table_name: str, data: List[Dict[str, Any]]) -> bool:
        """
        Truncates the table and inserts new data.
        Used for the 'Save' functionality after editing.
        """
        async with self.async_session_factory() as session:
            try:
                # Use TRUNCATE CASCADE to clear data (and potentially dependent data if configured)
                # Be careful with CASCADE in production, but for synthetic data it ensures a clean slate.
                logger.info(f"Truncating table {table_name}...")
                await session.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
                await session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Truncate failed for {table_name}: {e}")
                return False

        # Re-use insert logic
        return await self.insert_data(table_name, data)

    async def fetch_data(self, query: str) -> List[Dict[str, Any]]:
        """Executes a SELECT query and returns list of dictionaries."""
        async with self.async_session_factory() as session:
            try:
                result = await session.execute(text(query))
                rows = result.mappings().all()
                return [dict(row) for row in rows]
            except SQLAlchemyError as e:
                logger.error(f"Query execution failed: {e}")
                raise e
