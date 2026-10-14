import pandas as pd
import joblib
from pathlib import Path
from tempfile import NamedTemporaryFile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "students.csv"
MODEL_PATH = PROJECT_DIR / "student_ann.joblib"


def train_model(data_path=DATA_PATH, model_path=MODEL_PATH):
    """Train and save the ANN, returning the model and held-out accuracy."""
    data = pd.read_csv(data_path)
    X = data[["study_hours", "attendance", "previous_marks"]]
    y = data["passed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ann", MLPClassifier(
            hidden_layer_sizes=(8,),
            activation="relu",
            max_iter=2000,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    # Replace only a complete model so running sessions never load a partial file.
    model_path = Path(model_path)
    with NamedTemporaryFile(dir=model_path.parent, suffix=".joblib", delete=False) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        joblib.dump(model, temporary_path)
        temporary_path.replace(model_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return model, accuracy


if __name__ == "__main__":
    model, accuracy = train_model()
    print("ANN training completed")
    print("Test accuracy:", round(accuracy * 100, 2), "%")
    print(f"Model saved successfully: {MODEL_PATH}")
