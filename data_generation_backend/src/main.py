import os
import sys
import importlib
import logging
import vertexai
from fastapi import FastAPI
from fastapi.routing import APIRouter
from pathlib import Path

from lifespan import get_lifespan
from logging_config import setup_logging


setup_logging()


logger = logging.getLogger(__name__)


def import_routers(fast_api_app: FastAPI, directory: str):
    """
    Recursively searches for FastAPI routers in the given directory and dynamically imports them.
    Registers each router with the provided FastAPI app.

    Args:
        fast_api_app (FastAPI): The FastAPI app to register the routers with.
        directory (str): The directory to search for routers.
    """
    logger.info(f"Starting to import routers from directory: {directory}")

    for root, dirs, files in os.walk(directory):
        logger.debug(f"Scanning directory: {root}")
        for file in files:
            logger.debug(f"Checking file: {file}")

            # Skip irrelevant directories
            if root.endswith("__pycache__") or root.endswith(".venv"):
                logger.debug(f"Skipping directory: {root}")
                continue

            if file.endswith("endpoints.py"):
                # Construct the module name
                dirname = ".".join(root.split("/")[2:])
                filename = file.split(".py")[0]
                if dirname:
                    module_path = f"{dirname}.{filename}"
                else:
                    module_path = f"{filename}"

                logger.info(f"Trying to import module: {module_path}")

                try:
                    # Import the router module
                    router_module = importlib.import_module(module_path)
                    logger.debug(f"Module {module_path} imported successfully.")

                    # Find the router object in the module
                    router = next(
                        (obj for obj in router_module.__dict__.values() if isinstance(obj, APIRouter)),
                        None,
                    )

                    if router:
                        # Register the router with the FastAPI app
                        fast_api_app.include_router(router)
                        logger.info(f"Successfully registered router from {module_path} with FastAPI app.")
                    else:
                        logger.warning(f"No FastAPI router found in module: {module_path}")

                except Exception as e:
                    logger.error(f"Failed to import module {module_path}: {e}", exc_info=True)

    logger.info("Finished importing routers.")


app = FastAPI(
    title="Data Generation Backend",
    description="Data Generation Backend",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=get_lifespan,
)


curr_dir = Path(os.path.dirname(os.path.abspath(__file__)))
apps_dir = os.fspath(Path(curr_dir.parent).resolve())
sys.path.append(apps_dir)


print(f"Starting router import: {curr_dir=}, {apps_dir=}")
import_routers(app, apps_dir)
vertexai.init(project=os.getenv("GOOGLE_CLOUD_PROJECT"), location=os.getenv('LOCATION', 'us-central1'))
