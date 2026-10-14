import streamlit as st
import pandas as pd
import joblib
from train_model import MODEL_PATH, train_model


@st.cache_resource(max_entries=1, show_spinner="Preparing the prediction model...")
def load_model(model_path, modified_time):
    """Train a missing model and reload the cache after retraining."""
    if not model_path.is_file():
        model, _ = train_model(model_path=model_path)
        return model
    return joblib.load(model_path)

st.set_page_config(
    page_title="ANN Student Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")
st.write(
    "An Artificial Neural Network application "
    "for estimating student performance."
)

try:
    model_version = MODEL_PATH.stat().st_mtime_ns if MODEL_PATH.is_file() else None
    model = load_model(MODEL_PATH, model_version)
except (OSError, ValueError) as error:
    st.error(f"Could not prepare the prediction model: {error}")
    st.stop()

st.sidebar.header("Student Details")

study_hours = st.sidebar.slider(
    "Study hours per day",
    min_value=0,
    max_value=16,
    value=6
)

attendance = st.sidebar.slider(
    "Attendance percentage",
    min_value=0,
    max_value=100,
    value=85
)

previous_marks = st.sidebar.slider(
    "Previous exam marks",
    min_value=0,
    max_value=100,
    value=72
)

if st.sidebar.button("Predict Performance", type="primary"):

    student = pd.DataFrame(
        [[study_hours, attendance, previous_marks]],
        columns=[
            "study_hours",
            "attendance",
            "previous_marks"
        ]
    )

    prediction = model.predict(student)[0]

    probability = model.predict_proba(student)[0][1]

    st.subheader("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Estimated Pass Probability",
            f"{probability * 100:.1f}%"
        )

    with col2:
        st.metric(
            "Study Hours",
            f"{study_hours} hours/day"
        )

    if prediction == 1:
        st.success("Model prediction: Likely to Pass")
    else:
        st.warning(
            "Model prediction: May Need Additional Support"
        )

    st.subheader("Student Input Summary")

    summary = pd.DataFrame({
        "Parameter": [
            "Study Hours",
            "Attendance",
            "Previous Marks"
        ],
        "Value": [
            f"{study_hours} hours/day",
            f"{attendance}%",
            f"{previous_marks}/100"
        ]
    })

    st.dataframe(summary, hide_index=True)

    st.caption(
        "This is an educational demonstration using "
        "a small sample dataset, not a reliable assessment "
        "of an individual student's future performance."
    )
else:
    st.info(
        "Select the student details in the sidebar "
        "and click Predict Performance."
    )
