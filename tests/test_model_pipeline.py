"""Automated tests for model training pipeline, artifact serialization, and inference.

Tests run offline using small synthetic datasets to ensure complete test isolation.
"""

import os
import joblib
import pandas as pd
import pytest

from src.predict import load_model_pipeline, predict_email, SECURITY_DISCLAIMER
from src.train import (
    DEFAULT_TFIDF_PARAMS,
    train_and_evaluate_baseline_models,
)


@pytest.fixture
def sample_dataset(tmp_path):
    """Create a minimal clean dataset fixture for testing model training."""
    data = {
        "email_text": [
            "Quarterly budget spreadsheet attached for finance review",
            "Urgent: your online banking credentials require confirmation",
            "Team lunch meeting at the cafeteria at noon",
            "Security alert: unauthorized login detected verify account now",
            "Project roadmap discussion scheduled for tomorrow morning",
            "Congratulations you won a free cash reward click to claim",
            "Please review the revised marketing slide deck",
            "Notice: suspicious debit card transaction requires approval",
            "Weekly engineering standup minutes and action items",
            "Verify your corporate mailbox to prevent account deactivation",
        ] * 4,  # 40 rows
        "label": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 4,
    }
    csv_file = tmp_path / "test_clean_emails.csv"
    pd.DataFrame(data).to_csv(csv_file, index=False)
    return str(csv_file)


def test_train_and_evaluate_baseline_models(sample_dataset, tmp_path):
    """Verify end-to-end training, evaluation, and artifact saving."""
    models_dir = tmp_path / "models"
    results = train_and_evaluate_baseline_models(
        data_path=sample_dataset,
        models_dir=str(models_dir),
        results_filename="test_results.json",
        random_state=42,
        test_size=0.25,
    )

    # 1. Verify models exist in results dictionary
    assert "Logistic_Regression" in results["models"]
    assert "Random_Forest" in results["models"]

    # 2. Verify metric keys exist for each model
    for model_name, info in results["models"].items():
        metrics = info["metrics"]
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "false_positive_rate" in metrics
        assert "confusion_matrix" in metrics
        assert info["training_time_seconds"] >= 0.0
        assert info["evaluation_time_seconds"] >= 0.0

        # Verify artifacts exist on disk
        artifact = info["artifact_path"]
        assert os.path.exists(artifact)

    # 3. Verify machine-readable JSON exists
    json_path = models_dir / "test_results.json"
    assert os.path.exists(json_path)


def test_reproducibility_of_predictions(sample_dataset, tmp_path):
    """Verify that training with the same random_state yields identical test predictions."""
    models_dir1 = tmp_path / "models1"
    models_dir2 = tmp_path / "models2"

    res1 = train_and_evaluate_baseline_models(
        data_path=sample_dataset,
        models_dir=str(models_dir1),
        random_state=42,
    )
    res2 = train_and_evaluate_baseline_models(
        data_path=sample_dataset,
        models_dir=str(models_dir2),
        random_state=42,
    )

    # Predictions and metrics must be identical
    lr_metrics1 = res1["models"]["Logistic_Regression"]["metrics"]
    lr_metrics2 = res2["models"]["Logistic_Regression"]["metrics"]
    assert lr_metrics1 == lr_metrics2


def test_predict_email_inference(sample_dataset, tmp_path):
    """Verify inference output format, threshold adjustments, and security disclaimer."""
    models_dir = tmp_path / "models"
    train_and_evaluate_baseline_models(
        data_path=sample_dataset,
        models_dir=str(models_dir),
        random_state=42,
    )

    lr_artifact = str(models_dir / "logistic_regression_pipeline.joblib")
    pipeline = load_model_pipeline(lr_artifact)

    # Test valid prediction
    sample_text = "Important update: please confirm your password immediately"
    res = predict_email(sample_text, pipeline=pipeline)

    assert "prediction_label" in res
    assert res["prediction_label"] in ["Safe Email", "Phishing Email"]
    assert "phishing_probability" in res
    assert 0.0 <= res["phishing_probability"] <= 1.0
    assert res["security_disclaimer"] == SECURITY_DISCLAIMER

    # Test threshold effect
    res_high = predict_email(sample_text, pipeline=pipeline, threshold=0.99)
    # At 0.99 threshold, probability must be >= 0.99 to flag
    if res["phishing_probability"] < 0.99:
        assert res_high["prediction_label"] == "Safe Email"

    # Test edge cases
    with pytest.raises(ValueError, match="Cannot classify empty or blank"):
        predict_email("   ", pipeline=pipeline)

    with pytest.raises(ValueError, match="Decision threshold must be between"):
        predict_email(sample_text, pipeline=pipeline, threshold=1.5)


def test_load_model_missing_file_raises_error(tmp_path):
    """Verify that attempting to load a non-existent model artifact raises FileNotFoundError."""
    fake_path = str(tmp_path / "non_existent_model.joblib")
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        load_model_pipeline(fake_path)
