from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "train data.zip"
TRAIN_SCRIPT = ROOT / "training" / "train.py"

st.set_page_config(
    page_title="Customer Retention ML Lab",
    page_icon="📊",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_preview(path: str, rows: int = 2000) -> pd.DataFrame:
    return pd.read_csv(path, nrows=rows)


def run_experiment(data_path: str, min_recall: float):
    command = [
        sys.executable,
        "-m",
        "training.train",
        "--data",
        data_path,
        "--out",
        str(ROOT / "artifacts"),
        "--min-recall",
        str(min_recall),
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, result.stdout, result.stderr


st.title("Customer Retention ML Lab")
st.caption(
    "Streamlined experiment workspace for churn modeling, threshold tuning, "
    "evaluation, and production artifact generation."
)

with st.sidebar:
    st.header("Experiment")
    data_path = st.text_input(
        "Dataset path",
        str(DEFAULT_DATA),
    )
    min_recall = st.slider(
        "Minimum churn recall",
        min_value=0.50,
        max_value=0.99,
        value=0.80,
        step=0.01,
    )

    if st.button("Run full experiment", use_container_width=True):
        with st.spinner("Training and evaluating model..."):
            code, stdout, stderr = run_experiment(
                data_path,
                min_recall,
            )

        if code == 0:
            st.success("Experiment completed.")
            st.session_state["experiment_stdout"] = stdout
            st.session_state["experiment_stderr"] = stderr
        else:
            st.error("Experiment failed.")
            st.session_state["experiment_stdout"] = stdout
            st.session_state["experiment_stderr"] = stderr

tabs = st.tabs(
    [
        "Overview",
        "Data",
        "Experiment Output",
        "Artifacts",
        "API",
    ]
)

with tabs[0]:
    st.subheader("Pipeline")
    st.code(
        "Data → Preprocessing → Logistic Regression → "
        "Threshold Optimization → MLflow → Artifact → FastAPI",
        language="text",
    )

    metadata_path = ROOT / "artifacts" / "metadata.json"
    threshold_path = ROOT / "artifacts" / "threshold.json"

    if metadata_path.exists():
        import json

        metadata = json.loads(metadata_path.read_text())
        metrics = metadata.get("metrics", {})
        cols = st.columns(4)
        cols[0].metric("Recall", f'{metrics.get("recall", 0):.3f}')
        cols[1].metric("Precision", f'{metrics.get("precision", 0):.3f}')
        cols[2].metric("F1", f'{metrics.get("f1", 0):.3f}')
        cols[3].metric("PR-AUC", f'{metrics.get("pr_auc", 0):.3f}')
    else:
        st.info("Run an experiment to populate evaluation metrics.")

with tabs[1]:
    path = Path(data_path)
    if path.exists() and path.suffix.lower() in {".csv", ".zip"}:
        df = load_preview(str(path))
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows previewed", f"{len(df):,}")
        c2.metric("Columns", f"{len(df.columns):,}")
        c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.warning("Provide a local CSV path for the data preview.")

with tabs[2]:
    stdout = st.session_state.get("experiment_stdout")
    stderr = st.session_state.get("experiment_stderr")

    if stdout:
        st.subheader("Training output")
        st.code(stdout, language="text")
    if stderr:
        st.subheader("Warnings / errors")
        st.code(stderr, language="text")
    if not stdout and not stderr:
        st.info("No experiment has been run in this session.")

with tabs[3]:
    artifact_dir = ROOT / "artifacts"
    if artifact_dir.exists():
        files = [
            path.name
            for path in artifact_dir.iterdir()
            if path.is_file()
        ]
        st.write(files)
    else:
        st.info("No artifacts generated yet.")

with tabs[4]:
    st.subheader("Local API")
    st.write("Start FastAPI separately with:")
    st.code(
        "uvicorn api.main:app --host 0.0.0.0 --port 8000",
        language="bash",
    )
    st.write("Health: http://localhost:8000/health")\n    st.write("MLflow: http://localhost:5000")\n    st.write("Prometheus: http://localhost:9090")\n    st.write("Grafana: http://localhost:3000")
    st.write("Swagger: http://localhost:8000/docs")
