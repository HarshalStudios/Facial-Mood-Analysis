from __future__ import annotations
import argparse
from pathlib import Path
import cv2, joblib, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load(path, split):
    df=pd.read_csv(path); usage=df[' Usage'].astype(str).str.strip(); sub=df[usage==split]
    X=np.empty((len(sub),24*24),dtype=np.float32)
    for i,s in enumerate(sub[' pixels'].astype(str)):
        img=np.fromstring(s,sep=' ',dtype=np.uint8).reshape(48,48)
        X[i]=cv2.resize(img,(24,24),interpolation=cv2.INTER_AREA).ravel()/255.0
    return X, sub.emotion.to_numpy(np.int64)
p=argparse.ArgumentParser(); p.add_argument('--dataset-path',default='dataset/fer2013.csv'); p.add_argument('--output',default='models/fer2013_emotion_linear.joblib'); a=p.parse_args()
X,y=load(a.dataset_path,'Training'); VX,VY=load(a.dataset_path,'PublicTest')
pipe=Pipeline([('clf',LogisticRegression(C=1.0,max_iter=100,solver='lbfgs',class_weight='balanced'))])
pipe.fit(X,y); pred=pipe.predict(VX)
print('val_accuracy',accuracy_score(VY,pred)); print('val_macro_f1',f1_score(VY,pred,average='macro'))
# verify probabilities are finite on validation
probs=pipe.predict_proba(VX[:1000]); print('finite_probabilities',bool(np.isfinite(probs).all()),'row_sum_range',(probs.sum(1).min(),probs.sum(1).max()))
Path(a.output).parent.mkdir(parents=True,exist_ok=True); joblib.dump({'pipeline':pipe,'labels':list(range(7)),'input_shape':[24,24],'provider':'fer2013_logistic'},a.output); print('saved',a.output)
