from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, joblib, numpy as np, pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LABELS=np.arange(7)
def load(path):
    df=pd.read_csv(path)
    usage=df[' Usage'].astype(str).str.strip()
    out=[]
    for name,mask in [('train',usage=='Training'),('val',usage=='PublicTest')]:
        sub=df[mask]
        X=np.empty((len(sub),24*24),dtype=np.float32)
        for i,s in enumerate(sub[' pixels'].astype(str)):
            img=np.fromstring(s,sep=' ',dtype=np.uint8).reshape(48,48)
            X[i]=cv2.resize(img,(24,24),interpolation=cv2.INTER_AREA).ravel()/255.0
        out.append((X,sub.emotion.to_numpy(np.int64)))
    return out
p=argparse.ArgumentParser(); p.add_argument('--dataset-path',default='dataset/fer2013.csv'); p.add_argument('--output',default='models/fer2013_emotion_linear.joblib'); a=p.parse_args()
(trainX,trainy),(valX,valy)=load(a.dataset_path)
pipe=Pipeline([('scale',StandardScaler()),('clf',SGDClassifier(loss='log_loss',alpha=3e-5,max_iter=40,tol=1e-3,class_weight='balanced',random_state=42,early_stopping=True,validation_fraction=0.08,n_jobs=-1))])
pipe.fit(trainX,trainy)
pred=pipe.predict(valX)
print('val_accuracy',accuracy_score(valy,pred)); print('val_macro_f1',f1_score(valy,pred,average='macro'))
Path(a.output).parent.mkdir(parents=True,exist_ok=True)
joblib.dump({'pipeline':pipe,'labels':LABELS.tolist(),'input_shape':[24,24]},a.output)
print('saved',a.output)
