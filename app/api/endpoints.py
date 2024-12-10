"""
by: Jose Carlo Sia / chaoserver

FastAPI Endpoints for Chicken Vocalization Analysis API

This module defines the REST API endpoints for audio analysis.

Dependencies:
- fastapi: Web framework and API utilities
- uuid: For unique job ID generation
- aiofiles: Async file operations
- tempfile: Temporary file handling

Endpoints:
- POST /api/v1/analyze: Submit audio for analysis
- GET /api/v1/status/{job_id}: Check analysis status
- GET /api/v1/health: API health check
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from uuid import uuid4
import aiofiles
from pathlib import Path
import tempfile
import logging
from typing import Dict, Any

from app.core.inference import process_audio_file
from .models import AnalysisResult, JobStatus, JobResponse

# Configure router with version prefix
router = APIRouter(prefix="/api/v1")

# Store job results in memory (consider using Redis in production)
jobs: Dict[str, Dict[str, Any]] = {}

@router.post("/analyze", response_model=JobResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Submit audio file for distress call analysis.
    
    Process Flow:
    1. Generate unique job ID
    2. Save uploaded file to staging area
    3. Process audio asynchronously
    4. Return job ID for status checking
    
    Args:
        file: WAV format audio file
        
    Returns:
        JobResponse with unique job_id
        
    Raises:
        HTTPException: If file upload or processing fails
    """
    try:
        job_id = str(uuid4())
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = Path(temp_file.name)
        
        # Process audio asynchronously
        try:
            result = await process_audio_file(temp_path)
            jobs[job_id] = {
                "status": "completed",
                "result": result
            }
        except Exception as e:
            jobs[job_id] = {
                "status": "failed",
                "error": str(e)
            }
            raise
        finally:
            # Cleanup temporary file
            temp_path.unlink()
        
        return {"job_id": job_id}
        
    except Exception as e:
        logging.error(f"Error processing upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    Check the status of an analysis job.
    
    Args:
        job_id: Unique identifier from analyze endpoint
        
    Returns:
        JobStatus containing:
        - status: completed/failed
        - result: Analysis results if completed
        - error: Error message if failed
        
    Raises:
        HTTPException: If job_id not found
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@router.get("/health")
async def health_check():
    """
    Simple health check endpoint.
    
    Returns:
        Dict indicating API is operational
    """
    return {"status": "healthy"}

# Add other endpoints if needed 