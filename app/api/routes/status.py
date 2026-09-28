from fastapi import APIRouter

router = APIRouter(tags=["status"])

@router.get("/status")
def system_status():
    return {
        "status": "ready",
        "app_name": "Facial Mood Analysis",
        "version": "1.0.0",
        "model_version": "fusion_model_v1",
        "feature_schema_version": "1",
        "classes": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
        "dataset": "FER2013",
        "metrics": {
            "test_accuracy": 0.3761,
            "test_macro_f1": 0.3076
        }
    }

@router.get("/experiments")
def get_experiments():
    return [
        {"experiment": "Emotion-probability baseline", "test_accuracy": 0.3639, "test_macro_f1": 0.2972},
        {"experiment": "+ LBP", "test_accuracy": 0.3711, "test_macro_f1": 0.3022},
        {"experiment": "+ Edge", "test_accuracy": 0.3700, "test_macro_f1": 0.3027},
        {"experiment": "+ Gradient", "test_accuracy": 0.3697, "test_macro_f1": 0.3053},
        {"experiment": "+ Image Quality", "test_accuracy": 0.3703, "test_macro_f1": 0.3026},
        {"experiment": "+ All Handcrafted", "test_accuracy": 0.3761, "test_macro_f1": 0.3076},
        {"experiment": "Handcrafted Only", "test_accuracy": 0.2803, "test_macro_f1": 0.1746}
    ]
