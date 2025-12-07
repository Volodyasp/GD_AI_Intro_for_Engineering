import os

# Backend Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://data-generation-backend:8600")

# API Endpoints
GENERATE_ENDPOINT = f"{BACKEND_URL}/data-generation-engine/api/generate_data"
APPLY_CHANGE_ENDPOINT = f"{BACKEND_URL}/data-generation-engine/api/data-apply-change"
SAVE_DATA_ENDPOINT = f"{BACKEND_URL}/data-generation-engine/api/save_data"
TALK_ENDPOINT = f"{BACKEND_URL}/data-generation-engine/api/talk-to-data"
