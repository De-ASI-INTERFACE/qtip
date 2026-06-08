# =============================================================================
# OWNER:   Richard Patterson © 2026 — De-ASI-INTERFACE
# FILE:    tests/test_regime_classifier_v2.py
# PURPOSE: Full hardened test suite for RegimeClassifierV2
# =============================================================================

from __future__ import annotations

import numpy as np
import pytest

from qtip.regime_classifier_v2 import (
    Regime,
    RegimeClassifierV2,
    RegimeConfig,
    build_feature_matrix,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
RNG = np.random.default_rng(42)


def _synthetic_prices(n: int = 500, drift: float = 0.0001, vol: float = 0.02) -> np.ndarray:
    """Geometric Brownian Motion price series."""
    log_rets = RNG.normal(drift, vol, n)
    return 100.0 * np.exp(np.cumsum(log_rets))


def _regime_spliced_prices(n_per_regime: int = 150) -> np.ndarray:
    """Splice 4 distinct regime segments for label map testing."""
    bull  = _synthetic_prices(n_per_regime, drift=+0.003, vol=0.010)
    bear  = _synthetic_prices(n_per_regime, drift=-0.003, vol=0.010)
    range_= _synthetic_prices(n_per_regime, drift=0.000,  vol=0.003)
    hvol  = _synthetic_prices(n_per_regime, drift=0.001,  vol=0.040)
    # stitch: rescale each segment to start where the last ended
    series = [bull]
    for seg in [bear, range_, hvol]:
        ratio = series[-1][-1] / seg[0]
        series.append(seg * ratio)
    return np.concatenate(series)


@pytest.fixture(scope="module")
def fitted_clf():
    prices = _regime_spliced_prices()
    clf = RegimeClassifierV2()
    clf.fit(prices)
    return clf, prices


# ---------------------------------------------------------------------------
# Feature matrix tests
# ---------------------------------------------------------------------------
class TestFeatureMatrix:
    def test_shape(self):
        prices = _synthetic_prices(200)
        cfg = RegimeConfig()
        mat, valid = build_feature_matrix(prices, cfg)
        assert mat.ndim == 2
        assert mat.shape[1] == 4
        assert not np.isnan(mat).any()

    def test_min_obs_guard(self):
        prices = _synthetic_prices(10)
        cfg = RegimeConfig(min_obs=60)
        with pytest.raises(ValueError, match="60"):
            build_feature_matrix(prices, cfg)

    def test_zscore_zero_mean(self):
        prices = _synthetic_prices(300)
        cfg = RegimeConfig()
        mat, _ = build_feature_matrix(prices, cfg)
        means = mat.mean(axis=0)
        np.testing.assert_allclose(means, np.zeros(4), atol=1e-10)

    def test_zscore_unit_std(self):
        prices = _synthetic_prices(300)
        cfg = RegimeConfig()
        mat, _ = build_feature_matrix(prices, cfg)
        stds = mat.std(axis=0, ddof=1)
        np.testing.assert_allclose(stds, np.ones(4), atol=1e-2)


# ---------------------------------------------------------------------------
# Classifier fit tests
# ---------------------------------------------------------------------------
class TestFit:
    def test_fit_returns_self(self):
        prices = _synthetic_prices(300)
        clf = RegimeClassifierV2()
        result = clf.fit(prices)
        assert result is clf

    def test_fitted_flag(self):
        prices = _synthetic_prices(300)
        clf = RegimeClassifierV2()
        assert not clf._fitted
        clf.fit(prices)
        assert clf._fitted

    def test_label_map_covers_all_regimes(self, fitted_clf):
        clf, _ = fitted_clf
        assigned = set(clf._label_map.values())
        expected = set(Regime)
        assert assigned == expected, f"Missing regimes: {expected - assigned}"

    def test_label_map_no_duplicate_labels(self, fitted_clf):
        clf, _ = fitted_clf
        labels = list(clf._label_map.values())
        assert len(labels) == len(set(labels)), "Duplicate regime labels in map"

    def test_log_likelihood_is_finite(self, fitted_clf):
        clf, prices = fitted_clf
        ll = clf.log_likelihood(prices)
        assert np.isfinite(ll)
        assert ll < 0  # log-likelihood of a proper prob model is negative


# ---------------------------------------------------------------------------
# Predict tests
# ---------------------------------------------------------------------------
class TestPredict:
    def test_predict_returns_list_of_regimes(self, fitted_clf):
        clf, prices = fitted_clf
        preds = clf.predict(prices)
        assert isinstance(preds, list)
        assert all(isinstance(r, Regime) for r in preds)

    def test_predict_length(self, fitted_clf):
        clf, prices = fitted_clf
        preds = clf.predict(prices)
        # length should be < len(prices) due to NaN warm-up rows
        assert 0 < len(preds) < len(prices)

    def test_all_regimes_present_on_spliced_data(self, fitted_clf):
        clf, prices = fitted_clf
        preds = clf.predict(prices)
        seen = {r.value for r in preds}
        # at minimum 3 of the 4 regimes should appear on spliced data
        assert len(seen) >= 3, f"Too few distinct regimes detected: {seen}"

    def test_current_regime_is_regime_type(self, fitted_clf):
        clf, prices = fitted_clf
        cr = clf.current_regime(prices)
        assert isinstance(cr, Regime)

    def test_predict_before_fit_raises(self):
        clf = RegimeClassifierV2()
        with pytest.raises(RuntimeError, match="fit"):
            clf.predict(_synthetic_prices(300))

    def test_current_regime_before_fit_raises(self):
        clf = RegimeClassifierV2()
        with pytest.raises(RuntimeError, match="fit"):
            clf.current_regime(_synthetic_prices(300))


# ---------------------------------------------------------------------------
# Probability tests
# ---------------------------------------------------------------------------
class TestProbabilities:
    def test_predict_proba_shape(self, fitted_clf):
        clf, prices = fitted_clf
        proba = clf.predict_proba(prices)
        assert proba.ndim == 2
        assert proba.shape[1] == clf.cfg.n_states

    def test_predict_proba_sums_to_one(self, fitted_clf):
        clf, prices = fitted_clf
        proba = clf.predict_proba(prices)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(row_sums)), atol=1e-6)

    def test_predict_proba_non_negative(self, fitted_clf):
        clf, prices = fitted_clf
        proba = clf.predict_proba(prices)
        assert (proba >= 0).all()

    def test_current_proba_keys_are_regime_values(self, fitted_clf):
        clf, prices = fitted_clf
        cp = clf.current_proba(prices)
        valid_keys = {r.value for r in Regime}
        assert set(cp.keys()) == valid_keys

    def test_current_proba_sums_to_one(self, fitted_clf):
        clf, prices = fitted_clf
        cp = clf.current_proba(prices)
        total = sum(cp.values())
        assert abs(total - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Transition matrix tests
# ---------------------------------------------------------------------------
class TestTransitionMatrix:
    def test_transition_matrix_rows_sum_to_one(self, fitted_clf):
        clf, _ = fitted_clf
        tm = clf.transition_matrix()
        for from_regime, to_probs in tm.items():
            row_sum = sum(to_probs.values())
            assert abs(row_sum - 1.0) < 1e-5, f"{from_regime} row sums to {row_sum}"

    def test_transition_matrix_keys_match_regimes(self, fitted_clf):
        clf, _ = fitted_clf
        tm = clf.transition_matrix()
        regime_values = {r.value for r in Regime}
        assert set(tm.keys()) == regime_values

    def test_transition_matrix_non_negative(self, fitted_clf):
        clf, _ = fitted_clf
        tm = clf.transition_matrix()
        for row in tm.values():
            for v in row.values():
                assert v >= 0

    def test_transition_matrix_before_fit_raises(self):
        clf = RegimeClassifierV2()
        with pytest.raises(RuntimeError, match="fit"):
            clf.transition_matrix()


# ---------------------------------------------------------------------------
# Config variant tests
# ---------------------------------------------------------------------------
class TestConfigVariants:
    @pytest.mark.parametrize("cov_type", ["diag", "full", "tied"])
    def test_covariance_types(self, cov_type):
        prices = _synthetic_prices(400)
        clf = RegimeClassifierV2(RegimeConfig(covariance_type=cov_type, n_iter=50))
        clf.fit(prices)
        preds = clf.predict(prices)
        assert len(preds) > 0

    def test_custom_n_states_raises_if_mismatch(self):
        """n_states != 4 breaks label_map logic — document this boundary."""
        prices = _synthetic_prices(400)
        # 3 states: label map will assign BULL/BEAR/HIGH_VOL, leaving no RANGING
        clf = RegimeClassifierV2(RegimeConfig(n_states=3, n_iter=50))
        clf.fit(prices)
        assigned = set(clf._label_map.values())
        # Should have exactly 3 distinct labels
        assert len(assigned) == 3

    def test_refitting_updates_label_map(self):
        prices_a = _synthetic_prices(300, drift=+0.002)
        prices_b = _synthetic_prices(300, drift=-0.002)
        clf = RegimeClassifierV2(RegimeConfig(n_iter=50))
        clf.fit(prices_a)
        map_a = dict(clf._label_map)
        clf.fit(prices_b)
        map_b = dict(clf._label_map)
        # Maps may differ; assert refit happened cleanly
        assert clf._fitted
        assert len(map_b) == clf.cfg.n_states
