from app.experiments.registry import EXPERIMENTS, FEATURE_GROUPS, feature_names_for


def test_phase9_registry_has_expected_experiments():
    ids = [x.experiment_id for x in EXPERIMENTS]
    assert ids == [f"exp_{i:03d}" for i in range(1, 8)]


def test_feature_groups_match_canonical_schema():
    assert len(FEATURE_GROUPS["deep_emotion"]) == 7
    assert len(FEATURE_GROUPS["lbp"]) == 10
    assert len(FEATURE_GROUPS["edge"]) == 1
    assert len(FEATURE_GROUPS["gradient"]) == 1
    assert len(FEATURE_GROUPS["image_quality"]) == 3
    assert len(FEATURE_GROUPS["all_features"]) == 22


def test_feature_group_combination_deduplicates_in_order():
    names = feature_names_for(EXPERIMENTS[5])
    assert len(names) == 22
    assert names[0] == "emotion_angry"
    assert names[-1] == "sharpness"
