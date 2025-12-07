import logging
import os

logger = logging.getLogger(__name__)

# Import Langfuse v3
try:
    from langfuse import Langfuse, observe

    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    observe = None
    logger.warning("Langfuse SDK not installed. Observability disabled.")


class ObservabilityManager:
    """Singleton manager for Langfuse client (used for healthchecks)."""

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
        host = os.getenv("LANGFUSE_BASE_URL") or os.getenv(
            "LANGFUSE_HOST", "https://cloud.langfuse.com"
        )

        if LANGFUSE_AVAILABLE and public_key and secret_key:
            try:
                self.client = Langfuse(
                    public_key=public_key, secret_key=secret_key, host=host
                )
                logger.info(
                    f"Langfuse observability initialized successfully. Host: {host}"
                )
            except Exception as e:
                logger.error(f"Failed to initialize Langfuse: {e}")
        else:
            logger.warning(
                "Langfuse credentials not found or library missing. Observability disabled."
            )

    def check_health(self) -> dict:
        """Check Langfuse connection health."""
        if not LANGFUSE_AVAILABLE:
            return {
                "status": "unavailable",
                "message": "Langfuse library not installed",
            }

        if self.client is None:
            return {"status": "disabled", "message": "Langfuse not configured"}

        try:
            self.client.auth_check()
            return {"status": "healthy", "message": "Langfuse connection OK"}
        except Exception as e:
            return {"status": "unhealthy", "message": f"Langfuse auth failed: {str(e)}"}


def check_langfuse_health() -> dict:
    """Convenience function for healthcheck endpoints."""
    return ObservabilityManager().check_health()


# No-op decorator fallback
def _noop_observe(*args, **kwargs):
    """No-op decorator when Langfuse is not available."""

    def decorator(func):
        return func

    if len(args) == 1 and callable(args[0]):
        return args[0]
    return decorator


# Export the observe decorator directly - supports all parameters including as_type
if LANGFUSE_AVAILABLE and observe is not None:
    trace_step = observe
else:
    trace_step = _noop_observe
