"""Unit and integration tests for Streamlit application and prediction integration.

Verifies:
1. Saved model pipeline loading and structure.
2. Input validation (rejection of empty and whitespace-only text).
3. Prediction dictionary structure and mathematical probability sum.
4. Threshold effect adhering strictly to the 'probability >= threshold' rule.
5. Out-of-bounds threshold validation.
6. Immutability of vectorizer vocabulary (no retraining on user input).
7. Streamlit UI headless rendering, preset synchronization, input warning, and result cards.
"""

from unittest.mock import MagicMock
import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from streamlit.testing.v1 import AppTest

from app import (
    SAMPLE_EMAILS,
    get_cached_pipeline,
    set_threshold_preset,
)
from src.predict import (
    DEFAULT_MODEL_PATH,
    SECURITY_DISCLAIMER,
    load_model_pipeline,
    predict_email,
)


def test_saved_model_artifact_loads_and_has_expected_components():
    """Verify that the production Logistic Regression artifact loads and contains expected steps."""
    pipeline = load_model_pipeline(DEFAULT_MODEL_PATH)

    assert isinstance(pipeline, Pipeline)
    assert "tfidf" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps

    tfidf = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]

    # Verify TF-IDF is fitted with learned vocabulary
    assert hasattr(tfidf, "vocabulary_")
    assert len(tfidf.vocabulary_) > 1000

    # Verify classifier is Logistic Regression
    assert hasattr(classifier, "classes_")
    assert list(classifier.classes_) == [0, 1]


def test_prediction_empty_and_whitespace_input_rejected():
    """Verify that empty, blank, or non-string inputs raise ValueError."""
    with pytest.raises(ValueError, match="Cannot classify empty or blank"):
        predict_email("")

    with pytest.raises(ValueError, match="Cannot classify empty or blank"):
        predict_email("     \n\t   ")

    with pytest.raises(ValueError, match="Cannot classify empty or blank"):
        predict_email(None)  # type: ignore


def test_prediction_valid_text_output_structure():
    """Verify that valid email text produces all expected structured prediction fields."""
    sample_text = "Please review the updated quarterly budget forecast spreadsheet."
    res = predict_email(sample_text, threshold=0.60)

    assert "prediction_label" in res
    assert res["prediction_label"] in ["Safe Email", "Phishing Email"]
    assert "prediction_int" in res
    assert res["prediction_int"] in [0, 1]

    # Probabilities
    assert "phishing_probability" in res
    assert "safe_probability" in res
    assert 0.0 <= res["phishing_probability"] <= 1.0
    assert 0.0 <= res["safe_probability"] <= 1.0
    assert (res["phishing_probability"] + res["safe_probability"]) == pytest.approx(1.0, abs=1e-3)

    # Threshold and disclaimer
    assert res["decision_threshold"] == 0.60
    assert res["security_disclaimer"] == SECURITY_DISCLAIMER


def test_threshold_affects_classification_strictly_ge_rule():
    """Verify that the >= decision rule is applied consistently across threshold values."""
    # Create a mock pipeline returning fixed probabilities: P(Safe)=0.45, P(Phish)=0.55
    mock_pipeline = MagicMock(spec=Pipeline)
    mock_pipeline.predict_proba.return_value = np.array([[0.45, 0.55]])

    text = "Any sample email message body"

    # At threshold 0.50: 0.55 >= 0.50 -> Phishing Email
    res_50 = predict_email(text, pipeline=mock_pipeline, threshold=0.50)
    assert res_50["prediction_label"] == "Phishing Email"
    assert res_50["prediction_int"] == 1

    # At threshold 0.55 (exact boundary): 0.55 >= 0.55 -> Phishing Email
    res_55 = predict_email(text, pipeline=mock_pipeline, threshold=0.55)
    assert res_55["prediction_label"] == "Phishing Email"
    assert res_55["prediction_int"] == 1

    # At threshold 0.56: 0.55 < 0.56 -> Safe Email
    res_56 = predict_email(text, pipeline=mock_pipeline, threshold=0.56)
    assert res_56["prediction_label"] == "Safe Email"
    assert res_56["prediction_int"] == 0

    # At threshold 0.60: 0.55 < 0.60 -> Safe Email
    res_60 = predict_email(text, pipeline=mock_pipeline, threshold=0.60)
    assert res_60["prediction_label"] == "Safe Email"
    assert res_60["prediction_int"] == 0


def test_invalid_thresholds_rejected():
    """Verify that thresholds outside [0.0, 1.0] raise ValueError."""
    text = "Valid email message text"

    with pytest.raises(ValueError, match="Decision threshold must be between 0.0 and 1.0"):
        predict_email(text, threshold=-0.05)

    with pytest.raises(ValueError, match="Decision threshold must be between 0.0 and 1.0"):
        predict_email(text, threshold=1.05)


def test_no_retraining_on_user_input():
    """Verify that vectorizer vocabulary is immutable and does not learn unseen user words."""
    pipeline = load_model_pipeline(DEFAULT_MODEL_PATH)
    initial_vocab_len = len(pipeline.named_steps["tfidf"].vocabulary_)
    initial_vocab_keys = set(pipeline.named_steps["tfidf"].vocabulary_.keys())

    # User input containing completely novel, synthetic words
    novel_text = "XyzzyUnseenWord2026 FooBarBazMegaRandomString token12345"
    _ = predict_email(novel_text, pipeline=pipeline, threshold=0.60)

    # Vocabulary must remain 100% unchanged
    post_vocab_len = len(pipeline.named_steps["tfidf"].vocabulary_)
    post_vocab_keys = set(pipeline.named_steps["tfidf"].vocabulary_.keys())

    assert initial_vocab_len == post_vocab_len
    assert initial_vocab_keys == post_vocab_keys
    assert "xyzzyunseenword2026" not in post_vocab_keys


def test_streamlit_app_initialization_and_preset_sync():
    """Verify that app initializes with default threshold 0.60 and preset buttons update the slider."""
    at = AppTest.from_file("app.py").run(timeout=15)

    # Title check
    assert at.title[0].value == "Phishing Email Detection"

    # Default threshold check
    assert at.slider[0].value == 0.60

    # Preset button click: More sensitive (0.40)
    btn_sensitive = [b for b in at.button if "0.40" in b.label][0]
    btn_sensitive.click().run(timeout=15)
    assert at.slider[0].value == 0.40

    # Preset button click: Balanced (0.50)
    btn_balanced = [b for b in at.button if "0.50" in b.label][0]
    btn_balanced.click().run(timeout=15)
    assert at.slider[0].value == 0.50

    # Preset button click: Fewer false alarms (0.60)
    btn_low_fp = [b for b in at.button if "0.60" in b.label][0]
    btn_low_fp.click().run(timeout=15)
    assert at.slider[0].value == 0.60


def test_streamlit_app_empty_input_warning():
    """Verify that clicking Analyze Email with empty text surfaces an input warning."""
    at = AppTest.from_file("app.py").run(timeout=15)

    analyze_btn = [b for b in at.button if b.label == "Analyze Email"][0]
    analyze_btn.click().run(timeout=15)

    assert len(at.warning) > 0
    assert "Input Required" in at.warning[0].value


def test_streamlit_app_end_to_end_analysis_display():
    """Verify that submitting sample text renders category, probability, and decision explanations."""
    at = AppTest.from_file("app.py").run(timeout=15)

    # Populate email text
    phishing_sample = SAMPLE_EMAILS["Sample 2: Urgent Account Suspension Notice (Phishing)"]
    at.text_area[0].set_value(phishing_sample).run(timeout=15)

    analyze_btn = [b for b in at.button if b.label == "Analyze Email"][0]
    analyze_btn.click().run(timeout=15)

    # Verify no warning or error
    assert len(at.error) == 0

    # Verify results rendered
    assert any("Analysis Results" in sub.value for sub in at.subheader)
    assert any("Phishing Email" in m.value for m in at.markdown)
    assert len(at.metric) > 0
    assert "%" in at.metric[0].value
