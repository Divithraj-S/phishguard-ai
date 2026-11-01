"""Prediction and inference helper module for trained email classification models.

Loads serialized pipeline artifacts and evaluates arbitrary email text
with configurable probability thresholds and explicit security disclaimers.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Union

import joblib
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join("models", "logistic_regression_pipeline.joblib")
DEFAULT_THRESHOLD = 0.50

SECURITY_DISCLAIMER = (
    "This is an experimental text classifier. A low phishing score does not guarantee "
    "that an email is safe. Verify suspicious requests through trusted channels."
)


def load_model_pipeline(model_path: str = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a serialized scikit-learn Pipeline artifact.

    Args:
        model_path: Local filesystem path to the .joblib file.

    Returns:
        Loaded sklearn.pipeline.Pipeline object.

    Raises:
        FileNotFoundError: If the specified model file does not exist.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Please train models first using 'python -m src.train'."
        )
    return joblib.load(model_path)


def predict_email(
    email_text: str,
    pipeline: Optional[Pipeline] = None,
    model_path: str = DEFAULT_MODEL_PATH,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, Any]:
    """Classify an email text with a configurable decision threshold.

    Args:
        email_text: Raw or preprocessed email message string.
        pipeline: Pre-loaded sklearn.pipeline.Pipeline (optional).
        model_path: File path to load pipeline if not provided.
        threshold: Decision boundary probability (default 0.50).

    Returns:
        Structured prediction dictionary with label, probability, threshold, and disclaimer.

    Raises:
        ValueError: If input email_text is empty or threshold is outside [0.0, 1.0].
    """
    if not isinstance(email_text, str) or not email_text.strip():
        raise ValueError("Cannot classify empty or blank email text.")

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Decision threshold must be between 0.0 and 1.0. Got: {threshold}")

    if pipeline is None:
        pipeline = load_model_pipeline(model_path)

    # Predict probabilities: class 0 (Safe) vs class 1 (Phishing)
    probabilities = pipeline.predict_proba([email_text])[0]
    p_phish = float(probabilities[1])
    p_safe = float(probabilities[0])

    # Apply decision threshold
    is_phishing = int(p_phish >= threshold)
    label_name = "Phishing Email" if is_phishing == 1 else "Safe Email"

    return {
        "prediction_label": label_name,
        "prediction_int": is_phishing,
        "phishing_probability": round(p_phish, 4),
        "safe_probability": round(p_safe, 4),
        "decision_threshold": round(threshold, 2),
        "security_disclaimer": SECURITY_DISCLAIMER,
    }


if __name__ == "__main__":
    sample = "Urgent: Your bank account access has been suspended. Click here to confirm identity."
    print("Evaluating sample email text...")
    try:
        res = predict_email(sample)
        print(f"Result: {res['prediction_label']} (Phish Prob: {res['phishing_probability']:.2%})")
        print(f"Disclaimer: {res['security_disclaimer']}")
    except FileNotFoundError as err:
        print(err)
