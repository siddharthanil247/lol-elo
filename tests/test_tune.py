import numpy as np

from lol_elo.tune import _random_candidate


def test_random_search_role_weights_are_normalized_and_off_can_be_sampled() -> None:
    config = {
        "k_base": 32,
        "k_region": 0.1,
        "role_weights": {role: 1 for role in ("top", "jng", "mid", "bot", "sup")},
        "offseason_regression_enabled": False,
    }

    baseline = _random_candidate(np.random.default_rng(5), config, 0)
    random_candidate = _random_candidate(np.random.default_rng(7), config, 1)

    np.testing.assert_allclose(sum(baseline["role_weights"].values()), 1)
    np.testing.assert_allclose(sum(random_candidate["role_weights"].values()), 1)
    assert baseline["regress_frac"] == 0
    assert not baseline["offseason_regression_enabled"]
    assert 0 <= random_candidate["continuity_bonus"] <= 200
    assert 0 <= random_candidate["regress_frac"] <= 0.15
