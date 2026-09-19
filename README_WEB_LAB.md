# Customer Retention ML Lab

Run the complete experiment from a streamlined Streamlit workspace.

## Local setup

```bash
python -m pip install -r requirements.txt -r requirements-web.txt
streamlit run app.py
```

## What the web workspace does

- previews the selected dataset;
- runs the production training pipeline;
- tunes the churn decision threshold against a minimum recall target;
- displays generated metrics and experiment output;
- exposes generated model artifacts;
- links the local FastAPI inference service.

The web app intentionally executes the same `training.train` module used by CI, so the UI does not maintain a second copy of the training logic.
