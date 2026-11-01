"""Unit and integration tests for FastAPI backend (api.index).

Verifies:
1. Root endpoint metadata.
2. Health check endpoint reporting operational status and model readiness.
3. Successful email prediction with default threshold (0.60).
4. Prediction of legitimate safe email.
5. Configurable decision threshold adjustments (0.40 vs 0.80).
6. Validation: rejection of empty string via ValidationError.
7. Validation: rejection of whitespace-only string via ValidationError.
8. Validation: rejection of out-of-bounds threshold values (< 0.0 or > 1.0).
9. Model pipeline caching: verified that pipeline is loaded once and reused (singleton).
10. Error handling: simulation of missing model file returning HTTP 503.
"""

from unittest.mock import patch
import pytest
from pydantic import ValidationError
from fastapi import HTTPException

from api.index import (
    app,
    get_cached_pipeline,
    health_check,
    predict_endpoint,
    root,
    PredictRequest,
    PredictResponse,
    HealthResponse,
)
from src.predict import SECURITY_DISCLAIMER


def test_root_endpoint_metadata():
    """Verify that the root endpoint returns API identification metadata."""
    data = root()
    assert isinstance(data, dict)
    assert data["service"] == "PhishGuard AI API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "operational"
    assert "endpoints" in data
    assert data["endpoints"]["predict"] == "/api/predict"


def test_health_check_endpoint():
    """Verify that the health check endpoint reports operational status and loaded model."""
    res = health_check()
    assert isinstance(res, HealthResponse)
    assert res.status == "healthy"
    assert res.model_loaded is True
    assert res.default_threshold == 0.60
    assert "Logistic Regression" in res.model_name


def test_predict_endpoint_valid_phishing_email_default_threshold():
    """Verify prediction on a clear phishing email with default threshold (0.60)."""
    payload = PredictRequest(
        email_text="URGENT: Your online banking credentials have been suspended! Verify your account immediately.",
    )
    res = predict_endpoint(payload)

    assert isinstance(res, PredictResponse)
    assert res.prediction_label == "Phishing Email"
    assert res.prediction_int == 1
    assert 0.0 <= res.phishing_probability <= 1.0
    assert 0.0 <= res.safe_probability <= 1.0
    assert (res.phishing_probability + res.safe_probability) == pytest.approx(1.0, abs=1e-3)
    assert res.decision_threshold == 0.60
    assert res.security_disclaimer == SECURITY_DISCLAIMER


def test_predict_endpoint_valid_safe_email_default_threshold():
    """Verify prediction on a legitimate business email with default threshold (0.60)."""
    payload = PredictRequest(
        email_text="Hi Team, please find attached the quarterly meeting agenda and roadmap draft.",
    )
    res = predict_endpoint(payload)

    assert isinstance(res, PredictResponse)
    assert res.prediction_label == "Safe Email"
    assert res.prediction_int == 0
    assert res.phishing_probability < 0.60
    assert res.safe_probability > 0.40
    assert res.decision_threshold == 0.60


def test_predict_endpoint_custom_threshold_effect():
    """Verify that providing an explicit threshold changes the decision_threshold in the response."""
    email = "Dear customer, your invoice is ready for review. Please contact billing."

    # Test with sensitive threshold 0.40
    req_40 = PredictRequest(email_text=email, threshold=0.40)
    res_40 = predict_endpoint(req_40)
    assert res_40.decision_threshold == 0.40

    # Test with conservative threshold 0.80
    req_80 = PredictRequest(email_text=email, threshold=0.80)
    res_80 = predict_endpoint(req_80)
    assert res_80.decision_threshold == 0.80


def test_predict_empty_text_raises_validation_error():
    """Verify that an empty string is rejected by Pydantic input validation."""
    with pytest.raises(ValidationError):
        PredictRequest(email_text="")


def test_predict_whitespace_only_text_raises_validation_error():
    """Verify that whitespace-only text is rejected by Pydantic input validation."""
    with pytest.raises(ValidationError):
        PredictRequest(email_text="   \n\t   ")


def test_predict_invalid_threshold_raises_validation_error():
    """Verify that thresholds outside [0.0, 1.0] are rejected by Pydantic input validation."""
    # Negative threshold
    with pytest.raises(ValidationError):
        PredictRequest(email_text="Valid email message", threshold=-0.1)

    # Threshold > 1.0
    with pytest.raises(ValidationError):
        PredictRequest(email_text="Valid email message", threshold=1.5)


def test_model_pipeline_caching_singleton():
    """Verify that get_cached_pipeline returns the exact same object without reloading."""
    p1 = get_cached_pipeline()
    p2 = get_cached_pipeline()
    assert p1 is p2


def test_missing_model_artifact_returns_http_503():
    """Verify that if the model file is missing, the predict endpoint raises HTTP 503."""
    with patch("api.index.get_cached_pipeline", side_effect=FileNotFoundError("Model missing")):
        with pytest.raises(HTTPException) as exc_info:
            predict_endpoint(PredictRequest(email_text="Test email message body"))
        assert exc_info.value.status_code == 503
        assert "not available on the server" in exc_info.value.detail
