import os
import logging
import redis
from redis import Redis


logger = logging.getLogger(__name__)


class RedisSessionManager(Redis):
    _pool: redis.ConnectionPool | None = None

    def __init__(self):
        self.REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
        self.REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
        self.REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
        self.REDIS_DB = int(os.getenv('REDIS_DB', 0))

        self.initialize_pool()

    def initialize_pool(self):
        """
        Initializes the connection pool.
        This method should be called once on application startup.
        """
        if isinstance(self._pool, redis.ConnectionPool):
            logger.info("Redis connection pool already initialized")
            return None

        pool = redis.ConnectionPool(
            host=self.REDIS_HOST,
            port=self.REDIS_PORT,
            password=self.REDIS_PASSWORD or None,
            db=self.REDIS_DB,
            decode_responses=True,
            health_check_interval=30
        )
        type(self)._pool = pool
        return None

    def get_pool(self):
        if not self._is_pool_initialized():
            raise ConnectionError("Pool is not initialized, call initialize_pool() first")
        return type(self)._pool

    def _is_pool_initialized(self) -> bool:
        pool = type(self)._pool
        return pool is not None
