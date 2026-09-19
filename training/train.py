from __future__ import annotations
import argparse,json,os
from pathlib import Path
import joblib,mlflow,mlflow.sklearn
from sklearn.model_selection import train_test_split
from src.data import load_dataframe,prepare_xy
from src.evaluation import classification_metrics,optimize_threshold
from src.model import build_model

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",required=True)
    p.add_argument("--target",default=None)
    p.add_argument("--out",default="artifacts")
    p.add_argument("--min-recall",type=float,default=float(os.getenv("MIN_RECALL_TARGET","0.80")))
    a=p.parse_args()
    df=load_dataframe(a.data); X,y,target=prepare_xy(df,a.target)
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    model=build_model(Xtr)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI","http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT","customer-retention"))
    with mlflow.start_run() as run:
        model.fit(Xtr,ytr); prob=model.predict_proba(Xte)[:,1]
        default=classification_metrics(yte,prob,.5); opt=optimize_threshold(yte,prob,a.min_recall)
        mlflow.log_params({"model":"logistic_regression","class_weight":"balanced","random_state":42,"target_column":target,"min_recall_target":a.min_recall})
        mlflow.log_metrics({f"default_{k}":v for k,v in default.items()})
        mlflow.log_metrics({f"optimized_{k}":v for k,v in opt["metrics"].items()})
        mlflow.log_param("optimized_threshold",opt["threshold"])
        mlflow.sklearn.log_model(model,"model",registered_model_name=os.getenv("MLFLOW_REGISTERED_MODEL_NAME"))
        out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
        joblib.dump(model,out/"churn_model.joblib")
        (out/"threshold.json").write_text(json.dumps({"threshold":opt["threshold"],"min_recall_target":a.min_recall},indent=2))
        (out/"metadata.json").write_text(json.dumps({"run_id":run.info.run_id,"target_column":target,"feature_columns":list(X.columns),"metrics":opt["metrics"],"default_metrics":default},indent=2))
        print(json.dumps(opt,indent=2))
if __name__=="__main__": main()
