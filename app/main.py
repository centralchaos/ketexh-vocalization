from fastapi import FastAPI
from app.api.endpoints import router

# Create FastAPI app
app = FastAPI(
    title="Chicken Vocalization Analysis API",
    description="API for analyzing chicken vocalizations to detect distress calls",
    version="1.0.0"
)

# Include API routes
app.include_router(router) 