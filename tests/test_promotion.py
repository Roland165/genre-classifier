"""Règles de décision (ml/decide.py) : promotion et réentraînement, cas qui passent et échouent.

Métrique : macro F1, plus haut = meilleur. Les cas sont définis relativement au floor et au
seuil d'amélioration de ml/config.py, donc les tests restent valides si l'équipe change ces valeurs.
"""
import pytest

from ml import config
from ml.decide import should_promote, should_retrain

FLOOR = config.QUALITY_FLOOR
DELTA = config.MIN_IMPROVEMENT
CHAMPION_F1 = FLOOR + 0.20  # champion confortablement au-dessus du floor


# ---------- Promotion ----------

@pytest.mark.parametrize("candidate, champion, expected", [
    (CHAMPION_F1 + DELTA + 0.001, CHAMPION_F1, True),    # améliore de plus que le minimum
    (CHAMPION_F1 + 0.10, CHAMPION_F1, True),             # nettement meilleur
    (CHAMPION_F1 + DELTA / 2, CHAMPION_F1, False),       # amélioration trop faible
    (CHAMPION_F1, CHAMPION_F1, False),                   # égalité
    (CHAMPION_F1 - 0.05, CHAMPION_F1, False),            # moins bon
    (FLOOR - 0.01, FLOOR - 0.20, False),                 # meilleur que le champion mais sous le floor
    (FLOOR - 0.01, None, False),                         # première release sous le floor
    (FLOOR + 0.01, None, True),                          # première release au-dessus du floor
    (FLOOR, None, True),                                 # pile sur le floor : accepté
])
def test_should_promote(candidate, champion, expected):
    promoted, reason = should_promote(candidate, champion)
    assert promoted is expected
    assert reason


def test_promotion_reasons_name_the_failed_gate():
    assert "below quality floor" in should_promote(FLOOR - 0.01, None)[1]
    assert "not enough improvement" in should_promote(CHAMPION_F1 + DELTA / 2, CHAMPION_F1)[1]
    assert "better" in should_promote(CHAMPION_F1 + 0.10, CHAMPION_F1)[1]


def test_floor_is_checked_before_comparison_with_champion():
    # Un candidat sous le floor est refusé même si le champion est encore plus mauvais
    promoted, reason = should_promote(FLOOR - 0.01, FLOOR - 0.30)
    assert promoted is False
    assert "floor" in reason


# ---------- Réentraînement ----------

@pytest.mark.parametrize("champion_f1, champ_ds, new_ds, force, expected", [
    (None, None, "v1", False, True),                      # première release
    (CHAMPION_F1, "v1", "v2", False, True),               # nouvelle version de données
    (FLOOR - 0.05, "v1", "v1", False, True),              # champion sous le floor
    (CHAMPION_F1, "v1", "v1", False, False),              # mêmes données, champion correct
    (CHAMPION_F1, "v1", "v1", True, True),                # réentraînement forcé
])
def test_should_retrain(champion_f1, champ_ds, new_ds, force, expected):
    retrain, reason = should_retrain(champion_f1, champ_ds, new_ds, force=force)
    assert retrain is expected
    assert reason


def test_retrain_reasons():
    assert "first release" in should_retrain(None, None, "v1")[1]
    assert "v1 -> v2" in should_retrain(CHAMPION_F1, "v1", "v2")[1]
    assert "forced" in should_retrain(CHAMPION_F1, "v1", "v1", force=True)[1]
    assert "same dataset" in should_retrain(CHAMPION_F1, "v1", "v1")[1]