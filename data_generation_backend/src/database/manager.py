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
        host = os.getenv("POSTGRES_HOST", "postgres")
        port = os.getenv("POSTGRES_PORT", "5432")
        db_name = os.getenv("POSTGRES_DB", "postgres")

        self.database_url = f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"

        self.engine = create_async_engine(
            self.database_url,
            echo=False,
            future=True
        )

        self.async_session_factory = async_sessionmaker(
            self.engine,
            expire_on_commit=False,
            class_=AsyncSession
        )

    async def initialize_vector_db(self):
        """Enables pgvector extension and creates the examples table."""
        async with self.async_session_factory() as session:
            try:
                async with session.begin():
                    # 1. Enable Vector Extension
                    await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

                    # 2. Create Table for Few-Shot Examples
                    # We use 768 dimensions (standard for Vertex AI text-embedding-004)
                    await session.execute(text("""
                        CREATE TABLE IF NOT EXISTS sql_examples (
                            id SERIAL PRIMARY KEY,
                            question TEXT NOT NULL,
                            sql_query TEXT NOT NULL,
                            embedding vector(768)
                        );
                    """))
                logger.info("Vector DB (pgvector) initialized successfully.")
            except Exception as e:
                logger.error(f"Vector DB initialization failed: {e}")

    async def check_connection(self) -> bool:
        try:
            async with self.async_session_factory() as session:
                await session.execute(text("SELECT 1"))
            # Initialize vector setup on successful connection check
            await self.initialize_vector_db()
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False

    # --- Existing DDL/Data Methods ---
    async def execute_ddl(self, ddl_script: str) -> bool:
        async with self.async_session_factory() as session:
            try:
                async with session.begin():
                    statements = [s.strip() for s in ddl_script.split(';') if s.strip()]
                    for stmt in statements:
                        await session.execute(text(stmt))
                return True
            except SQLAlchemyError as e:
                logger.error(f"DDL Execution failed: {e}")
                raise e

    def _convert_value(self, value: Any) -> Any:
        if isinstance(value, str):
            if value.upper() == 'TRUE': return True
            if value.upper() == 'FALSE': return False
            if "T" in value or (":" in value and " " in value):
                try:
                    return datetime.fromisoformat(value)
                except ValueError:
                    pass
            if len(value) == 10 and value.count("-") == 2:
                try:
                    return date.fromisoformat(value)
                except ValueError:
                    pass
        return value

    async def insert_data(self, table_name: str, data: List[Dict[str, Any]]) -> bool:
        if not data: return False
        async with self.async_session_factory() as session:
            try:
                all_keys = set().union(*(d.keys() for d in data))
                normalized_data = []
                for row in data:
                    new_row = {k: self._convert_value(row.get(k, None)) for k in all_keys}
                    normalized_data.append(new_row)

                columns = ", ".join(all_keys)
                placeholders = ", ".join([f":{key}" for key in all_keys])
                sql = text(f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})")

                async with session.begin():
                    await session.execute(sql, normalized_data)
                return True
            except SQLAlchemyError as e:
                logger.error(f"Insert failed for {table_name}: {e}")
                raise e

    async def replace_data(self, table_name: str, data: List[Dict[str, Any]]) -> bool:
        async with self.async_session_factory() as session:
            try:
                await session.execute(text(f"TRUNCATE TABLE {table_name} CASCADE"))
                await session.commit()
            except SQLAlchemyError as e:
                logger.error(f"Truncate failed: {e}")
                return False
        return await self.insert_data(table_name, data)

    async def fetch_data(self, query: str) -> List[Dict[str, Any]]:
        async with self.async_session_factory() as session:
            try:
                result = await session.execute(text(query))
                rows = result.mappings().all()
                return [dict(row) for row in rows]
            except SQLAlchemyError as e:
                logger.error(f"Query execution failed: {e}")
                raise e

    async def insert_example(self, question: str, sql_query: str, embedding: List[float]):
        """Inserts a few-shot example with its embedding."""
        async with self.async_session_factory() as session:
            try:
                stmt = text("""
                    INSERT INTO sql_examples (question, sql_query, embedding)
                    VALUES (:q, :sql, :emb)
                """)
                # pgvector expects the embedding as a string representation of a list for raw SQL insertion usually,
                # but sqlalchemy+asyncpg handles list->vector conversion if the driver is set up right.
                # If issues arise, we might need to cast explicitly like `(:emb)::vector`
                await session.execute(stmt, {"q": question, "sql": sql_query, "emb": str(embedding)})
                await session.commit()
            except Exception as e:
                logger.error(f"Failed to insert example: {e}")

    async def search_similar_examples(self, query_embedding: List[float], limit: int = 3) -> List[Dict[str, str]]:
        """Finds the most similar SQL examples using Cosine Distance (<=>)."""
        async with self.async_session_factory() as session:
            try:
                # The <=> operator is cosine distance in pgvector
                stmt = text("""
                    SELECT question, sql_query
                    FROM sql_examples
                    ORDER BY embedding <=> :emb
                    LIMIT :lim
                """)
                # We pass the embedding as a string representation for pgvector compatibility in raw text queries
                result = await session.execute(stmt, {"emb": str(query_embedding), "lim": limit})
                return [{"question": row.question, "sql_query": row.sql_query} for row in result]
            except Exception as e:
                logger.error(f"Vector search failed: {e}")
                return []
