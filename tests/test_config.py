from app.core.config import get_settings


def test_settings_load_with_defaults():
    settings = get_settings()
    assert settings.app_name == "facial-mood-analysis-backend"
    assert settings.feature_vector_length == 22
    assert settings.api_port > 0


def test_settings_are_cached():
    a = get_settings()
    b = get_settings()
    assert a is b
