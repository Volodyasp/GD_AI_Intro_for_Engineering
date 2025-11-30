import os
import logging
from functools import wraps
from typing import Optional

# Import Langfuse
try:
    from langfuse import Langfuse
    from langfuse.decorators import observe

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

logger = logging.getLogger(__name__)


class ObservabilityManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ObservabilityManager, cls).__new__(cls)
            cls._instance.client = None
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        # Support both naming conventions, prioritizing the one in your .env
        host = os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

        if LANGFUSE_AVAILABLE and public_key and secret_key:
            try:
                self.client = Langfuse(
                    public_key=public_key,
                    secret_key=secret_key,
                    host=host
                )
                logger.info(f"Langfuse observability initialized successfully. Host: {host}")
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
        else:
            logger.warning("Langfuse credentials not found or library missing. Observability disabled.")


# Helper function to use as a decorator
def trace_step(name: Optional[str] = None):
    """
    Decorator to trace a function execution in Langfuse.
    If Langfuse is not configured, it simply runs the function.
    """

    def decorator(func):
        if not LANGFUSE_AVAILABLE:
            return func

        # If available, use the native Langfuse decorator
        # We wrap it to handle the 'name' parameter or default to function name
        @observe(name=name or func.__name__)
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        @observe(name=name or func.__name__)
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator
