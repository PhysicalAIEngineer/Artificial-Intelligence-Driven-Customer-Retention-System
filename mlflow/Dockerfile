FROM python:3.12-slim
WORKDIR /mlflow
RUN python -m pip install --no-cache-dir "mlflow>=2.18,<4" "psycopg2-binary>=2.9,<3"
EXPOSE 5000
CMD ["mlflow","server","--host","0.0.0.0","--port","5000","--backend-store-uri","postgresql+psycopg2://retention:retention@postgres:5432/retention","--default-artifact-root","/mlflow/artifacts","--allowed-hosts","*"]
