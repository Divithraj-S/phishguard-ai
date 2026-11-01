"""Streamlit Web Application for Phishing Email Detection.

Provides an interactive, beginner-friendly interface to evaluate email text
using a trained Logistic Regression model with TF-IDF features.

Key Capabilities:
- Interactive decision threshold slider (0.20 to 0.90, default 0.60).
- One-click presets: More sensitive (0.40), Balanced (0.50), Fewer false alarms (0.60).
- Transparent probability scores and decision explanations.
- Strict input validation and non-misleading, accessible styling.
- Prominent security disclaimers and scope limitations.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Dict, Optional

import streamlit as st

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.predict import (
    DEFAULT_MODEL_PATH,
    SECURITY_DISCLAIMER,
    load_model_pipeline,
    predict_email,
)

logger = logging.getLogger(__name__)

# Sample email texts for user testing and exploration
SAMPLE_EMAILS = {
    "-- Select a sample email (or paste your own below) --": "",
    "Sample 1: Legitimate Internal Meeting Notes (Safe)": (
        "Hi Team,\n\n"
        "Here are the minutes and action items from our weekly strategic planning session. "
        "Please review the attached quarterly goals draft before Thursday's standup. "
        "Let me know if you have any questions or updates to your departmental roadmaps.\n\n"
        "Best regards,\n"
        "Operations Team"
    ),
    "Sample 2: Urgent Account Suspension Notice (Phishing)": (
        "URGENT SECURITY ALERT:\n\n"
        "Your online banking access has been temporarily suspended due to multiple unauthorized "
        "login attempts from an unrecognized device. To prevent permanent deactivation, you must "
        "verify your credentials and confirm your identity immediately.\n\n"
        "Click here to update your security settings and restore account access.\n"
        "Failure to verify within 24 hours will result in permanent closure."
    ),
    "Sample 3: Borderline Expedited Invoice Notice": (
        "Dear Accounts Payable,\n\n"
        "Please find the revised invoice attached for professional services rendered last month. "
        "Kindly process the payment at your earliest convenience to maintain uninterrupted service. "
        "If you have already remitted payment, please disregard this automated notification.\n\n"
        "Thank you,\n"
        "Billing Department"
    ),
}


@st.cache_resource(show_spinner="Loading trained Logistic Regression model...")
def get_cached_pipeline(model_path: str = DEFAULT_MODEL_PATH) -> Any:
    """Load and cache the scikit-learn Pipeline artifact in memory.

    Args:
        model_path: Path to serialized .joblib pipeline file.

    Returns:
        Loaded sklearn Pipeline instance.
    """
    return load_model_pipeline(model_path=model_path)


def initialize_session_state() -> None:
    """Initialize Streamlit session state variables with safe defaults."""
    if "threshold" not in st.session_state:
        st.session_state["threshold"] = 0.60
    if "email_text" not in st.session_state:
        st.session_state["email_text"] = ""


def set_threshold_preset(value: float) -> None:
    """Callback to update the decision threshold in session state.

    Args:
        value: New probability threshold (e.g. 0.40, 0.50, 0.60).
    """
    st.session_state["threshold"] = round(float(value), 2)


def on_sample_email_selected() -> None:
    """Populate email input area when a sample email is picked from dropdown."""
    selected_key = st.session_state.get("sample_selector", "")
    sample_content = SAMPLE_EMAILS.get(selected_key, "")
    if sample_content:
        st.session_state["email_text"] = sample_content


def main() -> None:
    """Main application layout and execution flow."""
    st.set_page_config(
        page_title="Phishing Email Detection",
        layout="centered",
    )

    initialize_session_state()

    # Title & Application Description
    st.title("Phishing Email Detection")
    st.markdown(
        """
        Analyze email text using a trained **Logistic Regression with TF-IDF** model to estimate 
        whether the phrasing resembles legitimate correspondence or phishing attacks.
        """
    )

    # Prominent Disclaimer Card (Requirement 8)
    st.info(f"**Security Notice:** {SECURITY_DISCLAIMER}")

    # Explicit Scope and Capability Statement
    with st.expander("About This Tool & System Capabilities", expanded=False):
        st.markdown(
            """
            * **Text Pattern Analysis:** Analyzes vocabulary, phrasing, and text n-grams learned from email corpora.
            * **Scope Limitations:** This tool **does not** scan file attachments, inspect embedded images,
              browse live destination URLs, evaluate domain SPF/DKIM/DMARC headers, or verify sender identity.
            * **Model:** Evaluated using a reproducible, stratified pipeline trained on over 14,000 emails.
            * **No Retraining:** Vectorization uses a frozen vocabulary learned during training; user input is never used to retrain the model.
            """
        )

    # Pre-load or verify model artifact availability
    try:
        pipeline = get_cached_pipeline(DEFAULT_MODEL_PATH)
    except FileNotFoundError as err:
        st.error(
            f"**Model Artifact Missing:** {err}\n\n"
            "Please train the baseline models first by executing: "
            "`python -m src.train`"
        )
        return
    except Exception as exc:
        st.error(f"**Failed to load model pipeline:** {exc}")
        return

    st.subheader("1. Email Message Input")

    # Sample email selection helper
    st.selectbox(
        "Load a sample email (optional):",
        options=list(SAMPLE_EMAILS.keys()),
        index=0,
        key="sample_selector",
        on_change=on_sample_email_selected,
        help="Choose a pre-written sample to test the model, or paste your own message below.",
    )

    # Multiline text area for email content (Requirement 3)
    user_email_text = st.text_area(
        label="Paste email text to analyze (Subject and Body):",
        value=st.session_state["email_text"],
        height=180,
        placeholder="Paste complete email text here...",
        key="email_input_area",
    )
    # Keep session state updated with text changes
    st.session_state["email_text"] = user_email_text

    st.subheader("2. Decision Threshold Settings")
    st.markdown(
        """
        The **Decision Threshold** determines how confident the model must be before 
        classifying an email as **Phishing**. Adjusting this setting lets you control 
        the operational balance between catching threats and avoiding disruptive false alarms.
        """
    )

    # Preset Controls (Requirement 5)
    st.write("**Quick Presets:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button(
            "More sensitive (0.40)",
            on_click=set_threshold_preset,
            args=(0.40,),
            help="Flags more emails as phishing. Catches borderline threats, but causes more false alarms on safe emails.",
            use_container_width=True,
        )
    with col2:
        st.button(
            "Balanced (0.50)",
            on_click=set_threshold_preset,
            args=(0.50,),
            help="Standard default machine learning cut-off (50% probability).",
            use_container_width=True,
        )
    with col3:
        st.button(
            "Fewer false alarms (0.60)",
            on_click=set_threshold_preset,
            args=(0.60,),
            help="Recommended threshold from Phase 8 audit: cuts false alarms by ~42% while retaining 93.7% threat detection.",
            use_container_width=True,
        )

    # Decision-threshold slider (Requirement 4)
    chosen_threshold = st.slider(
        label="Decision Threshold:",
        min_value=0.20,
        max_value=0.90,
        value=st.session_state["threshold"],
        step=0.05,
        help="Emails with estimated Phishing Probability at or above this threshold are flagged as Phishing.",
    )
    st.session_state["threshold"] = chosen_threshold

    st.write(
        f"**Active Operating Threshold:** `{chosen_threshold:.2f}` "
        f"({chosen_threshold * 100:.0f}% confidence required to flag)"
    )

    # Analyze Button (Requirement 6)
    analyze_clicked = st.button("Analyze Email", type="primary", use_container_width=True)

    # Prediction Execution and Results Display (Requirement 7)
    if analyze_clicked:
        cleaned_input = user_email_text.strip() if user_email_text else ""

        if not cleaned_input:
            st.warning("Input Required: Please enter or paste an email message before analyzing.")
            return

        try:
            prediction = predict_email(
                email_text=cleaned_input,
                pipeline=pipeline,
                threshold=chosen_threshold,
            )
        except Exception as exc:
            st.error(f"An error occurred during email evaluation: {exc}")
            return

        label = prediction["prediction_label"]
        p_phish = prediction["phishing_probability"]
        p_safe = prediction["safe_probability"]
        p_phish_pct = p_phish * 100.0
        p_safe_pct = p_safe * 100.0
        threshold_pct = chosen_threshold * 100.0

        st.subheader("3. Analysis Results")

        # Neutral, accessible cards avoiding misleading bright green/red color-coding
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            st.markdown(f"**Predicted Category:**")
            if label == "Phishing Email":
                st.markdown(
                    f"### [Suspicious] **{label}**\n"
                    f"*Estimated phishing risk ({p_phish_pct:.1f}%) meets or exceeds "
                    f"the decision threshold ({threshold_pct:.0f}%).*"
                )
            else:
                st.markdown(
                    f"### [Likely Benign] **{label}**\n"
                    f"*Estimated phishing risk ({p_phish_pct:.1f}%) is below "
                    f"the decision threshold ({threshold_pct:.0f}%).*"
                )

        with res_col2:
            st.metric(
                label="Model Phishing Probability",
                value=f"{p_phish_pct:.2f}%",
                delta=f"{p_phish_pct - threshold_pct:+.1f}% vs threshold",
                delta_color="off",
            )
            st.write(f"Safe Email Probability: `{p_safe_pct:.2f}%`")

        # Visual Probability Bar
        st.write("**Probability Spectrum:**")
        st.progress(min(max(p_phish, 0.0), 1.0))
        st.caption(
            f"0% (Benign) ──── Threshold: {threshold_pct:.0f}% ──── 100% (High Suspicion). "
            f"Current Score: {p_phish_pct:.1f}%"
        )

        # Pedagogical Explanation of Decision Threshold Impact (Requirement 7)
        st.markdown("---")
        st.markdown("#### Understanding the Decision Rule")

        if p_phish >= chosen_threshold:
            rule_explanation = (
                f"Because the predicted phishing probability (**{p_phish_pct:.2f}%**) is **greater than or equal to** "
                f"the active decision threshold (**{threshold_pct:.0f}%**), this email is classified as **Phishing Email**."
            )
        else:
            rule_explanation = (
                f"Because the predicted phishing probability (**{p_phish_pct:.2f}%**) is **lower than** "
                f"the active decision threshold (**{threshold_pct:.0f}%**), this email is classified as **Safe Email**."
            )

        st.markdown(rule_explanation)

        st.markdown(
            f"""
            **Why changing the threshold can alter this prediction:**
            * A probabilistic model estimates *relative likelihood*, not an absolute guarantee.
            * If this email had received a probability of **55%**:
              - Under the **Balanced** setting (0.50), it would be flagged as **Phishing Email** (55% >= 50%).
              - Under the **Fewer false alarms** setting (0.60), it would be marked as **Safe Email** (55% < 60%).
            * **Lower thresholds** catch more attacks but increase false alarms on legitimate communications.
            * **Higher thresholds** prevent false alarms on business emails but require stronger indicators before flagging.
            """
        )

        # Permanent Footer Disclaimer
        st.markdown("---")
        st.caption(f"Security Verification Reminder: {SECURITY_DISCLAIMER}")


if __name__ == "__main__":
    main()
