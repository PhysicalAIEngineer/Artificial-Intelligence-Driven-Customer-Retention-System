import numpy as np
from src.evaluation import classification_metrics,optimize_threshold

def test_metrics():
    y=np.array([0,1,1,0]); p=np.array([.1,.9,.8,.2])
    assert classification_metrics(y,p,.5)["recall"]==1.0

def test_threshold():
    y=np.array([0,0,1,1,1]); p=np.array([.1,.2,.4,.8,.9])
    assert optimize_threshold(y,p,.8)["metrics"]["recall"]>=.8
