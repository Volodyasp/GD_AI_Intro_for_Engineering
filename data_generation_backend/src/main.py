from fastapi import FastAPI

from lifespan import get_lifespan


app = FastAPI(
    title="Data Generation Backend",
    description="Data Generation Backend",
    lifespan=get_lifespan,
)
