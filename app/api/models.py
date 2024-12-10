"""
by: Jose Carlo Sia / chaoserver

Pydantic Models for API Request/Response Handling

This module defines the data models used in the API endpoints for:
- Request validation
- Response serialization
- Type safety

Dependencies:
- pydantic: Data validation using Python type annotations
- typing: Type hints for better code documentation

Models:
- JobResponse: Initial response after submitting audio
- JobStatus: Status check response with results
- AnalysisResult: Detailed analysis results
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Union


class JobResponse(BaseModel):
    """
    Response model for /analyze endpoint
    
    Attributes:
        job_id: Unique identifier for tracking analysis progress
    """
    job_id: str = Field(..., description="Unique identifier for the analysis job")


class AnalysisResult(BaseModel):
    """
    Detailed results from audio analysis
    
    Attributes:
        total_segments: Total number of audio segments analyzed
        barn_segments: Number of segments classified as normal barn sounds
        distress_segments: Number of segments classified as distress calls
        distress_percentage: Percentage of segments containing distress calls
        average_confidence: Model's average confidence in predictions
        alert_level: HIGH/MODERATE/LOW based on distress percentage
        recommendation: Action recommendation based on alert level
        evaluation_metrics: Optional performance metrics
    """
    total_segments: int
    barn_segments: int
    distress_segments: int
    distress_percentage: float
    average_confidence: float
    alert_level: str
    recommendation: str
    evaluation_metrics: Optional[Dict[str, float]] = None


class JobStatus(BaseModel):
    """
    Response model for /status endpoint
    
    Attributes:
        status: Current job status (completed/failed)
        result: Analysis results if completed
        error: Error message if failed
    """
    status: str = Field(..., description="Job status: completed or failed")
    result: Optional[AnalysisResult] = Field(None, description="Analysis results if completed")
    error: Optional[str] = Field(None, description="Error message if failed") 