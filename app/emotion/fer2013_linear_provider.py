
from __future__ import annotations
from pathlib import Path
import cv2, joblib, numpy as np
from app.emotion.base import EmotionModel
from app.emotion.validation import CANONICAL_EMOTION_LABELS

class FER2013LinearEmotionModel(EmotionModel):
    """Real FER2013-trained probability provider using a supervised linear classifier.

    This is an explicit non-DeepFace backend. It is used when DeepFace is unavailable;
    probabilities are learned from the supplied FER2013 training split.
    """
    def __init__(self, model_path: str | Path):
        artifact=joblib.load(model_path)
        self.pipeline=artifact["pipeline"]
        self.labels=artifact.get("labels", list(range(7)))

    def predict_probabilities(self, face_crop: np.ndarray) -> dict[str,float]:
        arr=np.asarray(face_crop)
        if arr.ndim==3: arr=cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        if arr.ndim!=2: raise ValueError(f"Expected face image, got {arr.shape}")
        img=cv2.resize(arr.astype(np.uint8),(24,24),interpolation=cv2.INTER_AREA)
        x=(img.astype(np.float32)/255.0).reshape(1,-1)
        probs=self.pipeline.predict_proba(x)[0]
        result={label:0.0 for label in CANONICAL_EMOTION_LABELS}
        for cls,p in zip(self.pipeline.classes_,probs):
            result[CANONICAL_EMOTION_LABELS[int(cls)]]=float(p)
        return result
