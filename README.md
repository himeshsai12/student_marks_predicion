# student_marks_predicion

A Streamlit demonstration that uses an artificial neural network to estimate
whether a student will pass from study hours, attendance, and previous marks.

From the cloned project folder, create a Python environment, install the
dependencies, and start the app in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

For the existing local `uk` environment, run
`.\uk\Scripts\python.exe -m streamlit run app.py` instead.

The app automatically trains and saves `student_ann.joblib` beside the scripts
if the model is missing. To retrain after changing `students.csv`, run:

```powershell
.\.venv\Scripts\python.exe train_model.py
```

The scripts find their data and model relative to their own location, even when
launched from another working directory. Refresh an already open browser tab to
load the latest app.

The dataset contains only 20 sample students, with five used for testing. Model
probabilities and test accuracy are demonstration results, not reliable measures
of a student's future performance.
