"""
Retraining and promotion rules.
Pure functions (no MLflow) so they are easy to unit test.
Metric = macro F1, higher is better.
"""
from ml import config


def should_retrain(champion_f1, champion_dataset, new_dataset,
                   floor=config.QUALITY_FLOOR, force=False):
    """Return (True/False, reason)."""
    if force:
        return True, "retraining forced by user"
    if champion_f1 is None:
        return True, "no champion yet (first release)"
    if champion_dataset != new_dataset:
        return True, f"new dataset version ({champion_dataset} -> {new_dataset})"
    if champion_f1 < floor:
        return True, f"champion below quality floor ({champion_f1:.4f} < {floor})"
    return False, "same dataset and champion still above the floor"


def should_promote(candidate_f1, champion_f1,
                   floor=config.QUALITY_FLOOR, min_improvement=config.MIN_IMPROVEMENT):
    """Return (True/False, reason)."""
    if candidate_f1 < floor:
        return False, f"candidate below quality floor ({candidate_f1:.4f} < {floor})"
    if champion_f1 is None:
        return True, "first release and candidate above the floor"
    if candidate_f1 < champion_f1 + min_improvement:
        return False, (f"not enough improvement ({candidate_f1:.4f} vs champion "
                       f"{champion_f1:.4f}, need +{min_improvement})")
    return True, f"candidate is better ({candidate_f1:.4f} vs {champion_f1:.4f})"
