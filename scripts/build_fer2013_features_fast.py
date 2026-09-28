
from __future__ import annotations
import argparse, csv, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2, numpy as np, pandas as pd
from skimage.feature import local_binary_pattern
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.core.config import get_settings
from app.schemas.feature_vector import FEATURE_NAMES
from app.emotion.factory import get_emotion_model

def one(s, settings):
    img=np.fromstring(s,sep=' ',dtype=np.uint8).reshape(48,48)
    gray=img
    clahe=cv2.createCLAHE(clipLimit=settings.clahe_clip_limit,tileGridSize=(settings.clahe_tile_grid_size,settings.clahe_tile_grid_size)).apply(gray)
    canny=cv2.Canny(gray,settings.canny_low_threshold,settings.canny_high_threshold,apertureSize=settings.canny_aperture_size,L2gradient=settings.canny_l2_gradient)
    lbp=local_binary_pattern(gray,P=settings.lbp_n_points,R=settings.lbp_radius,method=settings.lbp_method).astype(np.uint8)
    hist,_=np.histogram(lbp,bins=10,range=(0,10)); hist=hist.astype(np.float64)/hist.sum()
    edge=float(np.count_nonzero(canny))/canny.size
    gf=gray.astype(np.float64)/255.0
    gx=cv2.Sobel(gf,cv2.CV_64F,1,0,ksize=settings.sobel_kernel_size); gy=cv2.Sobel(gf,cv2.CV_64F,0,1,ksize=settings.sobel_kernel_size)
    grad=float(np.mean(gx*gx+gy*gy))
    bright=float(np.mean(gray.astype(np.float64))); contrast=float(np.std(gray.astype(np.float64)))
    sharp=float(np.var(cv2.Laplacian(gray,cv2.CV_64F)))
    x=cv2.resize(img,(24,24),interpolation=cv2.INTER_AREA).astype(np.float32).reshape(-1)/255.0
    return x,hist,edge,grad,bright,contrast,sharp

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset-path',default='dataset/fer2013.csv'); ap.add_argument('--emotion-backend',default='fer2013_linear'); ap.add_argument('--workers',type=int,default=8); args=ap.parse_args()
    settings=get_settings().model_copy(update={'emotion_model_backend':args.emotion_backend})
    model=get_emotion_model(settings)
    if not hasattr(model,'pipeline'):
        raise RuntimeError('fast builder requires a batch-capable FER2013 linear provider')
    df=pd.read_csv(args.dataset_path); usage=df[' Usage'].astype(str).str.strip()
    outdir=Path(settings.features_dir)/'raw'; outdir.mkdir(parents=True,exist_ok=True)
    fields=['sample_id','split','emotion','emotion_id',*FEATURE_NAMES]
    for split,mask in [('train',usage=='Training'),('validation',usage=='PublicTest'),('test',usage=='PrivateTest')]:
        sub=df[mask].reset_index(drop=True); path=outdir/f'{split}_features.csv'
        existing = 0
        if path.exists():
            try:
                existing = max(0, sum(1 for _ in open(path, encoding='utf-8')) - 1)
            except Exception:
                existing = 0
        rows=[]
        if existing:
            print('resume',split,'starting_at',existing,'of',len(sub),flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for start in range(existing,len(sub),256):
                chunk=sub.iloc[start:start+256]
                vals=list(ex.map(lambda s: one(s,settings), chunk[' pixels'].astype(str)))
                X=np.stack([v[0] for v in vals])
                probs=model.pipeline.predict_proba(X)
                for j,v in enumerate(vals):
                    _,hist,edge,grad,bright,contrast,sharp=v
                    feat=np.concatenate([probs[j],hist,[edge,grad,bright,contrast,sharp]])
                    rows.append([str(chunk.iloc[j].name),split,int(chunk.iloc[j].emotion),int(chunk.iloc[j].emotion),*feat.tolist()])
                if len(rows)>=2000:
                    with open(path,'a',newline='',encoding='utf-8') as f:
                        w=csv.writer(f)
                        if f.tell()==0: w.writerow(fields)
                        w.writerows(rows)
                    rows=[]
                if (start+len(chunk))%2000 < 256: print(split,start+len(chunk),len(sub),flush=True)
        if rows:
            with open(path,'a',newline='',encoding='utf-8') as f:
                w=csv.writer(f)
                if f.tell()==0:w.writerow(fields)
                w.writerows(rows)
        print('done',split,path,flush=True)

if __name__=='__main__':main()
