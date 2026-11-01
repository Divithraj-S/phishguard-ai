"""FastAPI Backend Application for PhishGuard AI.

Provides REST API endpoints for phishing email detection using the pre-trained
Logistic Regression + TF-IDF model pipeline.

Designed for both local execution (`uvicorn api.index:app`) and serverless
deployment on Vercel (`/api/index.py`).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import os
import sys
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Ensure project root is available in Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import (
    DEFAULT_MODEL_PATH,
    SECURITY_DISCLAIMER,
    load_model_pipeline,
    predict_email,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Resolve absolute model path so it works regardless of working directory
RESOLVED_MODEL_PATH = (
    DEFAULT_MODEL_PATH
    if os.path.isabs(DEFAULT_MODEL_PATH)
    else os.path.join(PROJECT_ROOT, DEFAULT_MODEL_PATH)
)

# Global cached model pipeline instance to prevent reloading on every request
_cached_pipeline: Optional[Any] = None


def get_cached_pipeline(model_path: str = RESOLVED_MODEL_PATH) -> Any:
    """Retrieve the cached model pipeline or load it once into memory.

    Args:
        model_path: Absolute or relative path to the .joblib artifact.

    Returns:
        The loaded scikit-learn Pipeline.

    Raises:
        FileNotFoundError: If the model artifact cannot be found.
    """
    global _cached_pipeline
    if _cached_pipeline is None:
        if not os.path.exists(model_path):
            logger.error("Model artifact missing at resolved path: '%s'", model_path)
            raise FileNotFoundError(
                f"Model artifact not found at '{model_path}'. "
                "Please ensure models/logistic_regression_pipeline.joblib is present."
            )
        logger.info("Loading model pipeline into memory from '%s'...", model_path)
        _cached_pipeline = load_model_pipeline(model_path)
        logger.info("Model pipeline loaded successfully.")
    return _cached_pipeline


app = FastAPI(
    title="PhishGuard AI API",
    description="Machine Learning REST API for Phishing Email Detection with configurable decision thresholds.",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

# Enable CORS for local Next.js frontend (localhost:3000) and Vercel domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# --- Request and Response Schemas ---

class PredictRequest(BaseModel):
    """Input payload for email classification."""

    email_text: str = Field(
        ...,
        description="The raw email text (subject and/or body) to analyze.",
        examples=[
            "Urgent: Your bank account access has been suspended. Click here to confirm identity."
        ],
    )
    threshold: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Decision threshold probability (default: 0.60, recommended range: 0.20 to 0.90).",
        examples=[0.60],
    )

    @field_validator("email_text")
    @classmethod
    def validate_non_empty_text(cls, value: str) -> str:
        """Ensure input text is not empty or whitespace-only."""
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Email text cannot be empty or whitespace only.")
        return value.strip()

    @field_validator("threshold")
    @classmethod
    def validate_threshold_range(cls, value: float) -> float:
        """Ensure threshold is within [0.0, 1.0]."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"Decision threshold must be between 0.0 and 1.0. Got: {value}")
        return round(float(value), 4)


class PredictResponse(BaseModel):
    """Prediction output schema."""

    prediction_label: str = Field(
        ...,
        description="Classification label: 'Phishing Email' or 'Safe Email'.",
        examples=["Phishing Email"],
    )
    prediction_int: int = Field(
        ...,
        description="Binary prediction code (1 for Phishing, 0 for Safe).",
        examples=[1],
    )
    phishing_probability: float = Field(
        ...,
        description="Estimated probability that the email is phishing (0.0 to 1.0).",
        examples=[0.9842],
    )
    safe_probability: float = Field(
        ...,
        description="Estimated probability that the email is safe (0.0 to 1.0).",
        examples=[0.0158],
    )
    decision_threshold: float = Field(
        ...,
        description="Decision threshold applied to the prediction.",
        examples=[0.60],
    )
    security_disclaimer: str = Field(
        ...,
        description="Mandatory disclaimer about experimental machine learning predictions.",
        examples=[SECURITY_DISCLAIMER],
    )


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    model_loaded: bool
    model_path: str
    default_threshold: float
    model_name: str


# --- API Routes ---

@app.get("/", tags=["General"])
@app.get("/api", tags=["General"])
def root() -> Dict[str, Any]:
    """Root endpoint providing service information."""
    return {
        "service": "PhishGuard AI API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict",
            "documentation": "/api/docs",
        },
    }


@app.get("/api/health", response_model=HealthResponse, tags=["Monitoring"])
@app.get("/health", response_model=HealthResponse, include_in_schema=False)
def health_check() -> HealthResponse:
    """Check system health and model availability."""
    try:
        pipeline = get_cached_pipeline()
        loaded = pipeline is not None
    except Exception:
        loaded = False

    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        model_path=os.path.relpath(RESOLVED_MODEL_PATH, PROJECT_ROOT),
        default_threshold=0.60,
        model_name="Logistic Regression with TF-IDF (10,000 features)",
    )


@app.post(
    "/api/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
)
@app.post(
    "/predict",
    response_model=PredictResponse,
    status_code=status.HTTP_200_OK,
    include_in_schema=False,
)
def predict_endpoint(payload: PredictRequest) -> PredictResponse:
    """Analyze an email message and return phishing risk assessment.

    Args:
        payload: PredictRequest containing email_text and optional decision threshold.

    Returns:
        PredictResponse with classification label, probabilities, and disclaimer.

    Raises:
        HTTPException 503: If the trained model artifact is missing.
        HTTPException 400: If prediction fails on input.
    """
    try:
        pipeline = get_cached_pipeline()
    except FileNotFoundError as err:
        logger.error("Model artifact not found: %s", err)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Trained model artifact is not available on the server. Please contact administrator.",
        )
    except Exception as exc:
        logger.error("Unexpected error loading model: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load prediction pipeline: {exc}",
        )

    try:
        # Reuses existing prediction logic from src/predict.py
        result = predict_email(
            email_text=payload.email_text,
            pipeline=pipeline,
            threshold=payload.threshold,
        )
        return PredictResponse(**result)
    except ValueError as err:
        logger.warning("Input validation error in predict_email: %s", err)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(err),
        )
    except Exception as exc:
        logger.error("Inference failure: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred during email inference: {exc}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.index:app", host="127.0.0.1", port=8000, reload=True)
