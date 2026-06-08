# =============================================================================
# OWNER:   Richard Patterson © 2026 — De-ASI-INTERFACE
# FILE:    qtip/regime_classifier_v2.py
# PURPOSE: Regime Classifier v2 — HMM-based market state transitions
#          States: BULL_TREND | BEAR_TREND | RANGING | HIGH_VOL_BREAKOUT
# VERSION: 2.0.0
# =============================================================================

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
from hmmlearn import hmm
from scipy.stats import zscore

warnings.filterwarnings("ignore", category=DeprecationWarning)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regime Enum
# ---------------------------------------------------------------------------
class Regime(str, Enum):
    BULL_TREND       = "BULL_TREND"
    BEAR_TREND       = "BEAR_TREND"
    RANGING          = "RANGING"
    HIGH_VOL_BREAKOUT = "HIGH_VOL_BREAKOUT"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class RegimeConfig:
    n_states:          int   = 4          # must match len(Regime)
    n_iter:            int   = 200        # EM iterations
    covariance_type:   str   = "full"     # full | diag | tied | spherical
    tol:               float = 1e-5
    random_state:      int   = 42
    min_obs:           int   = 60         # minimum bars required before fit
    zscore_window:     int   = 20         # rolling window for feature z-score
    vol_lookback:      int   = 14         # ATR-style vol lookback
    momentum_lookback: int   = 10         # ROC lookback
    # State-to-regime label mapping (tuned post-fit by mean return rank)
    label_map: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------
def _returns(prices: np.ndarray) -> np.ndarray:
    """Log returns."""
    return np.diff(np.log(prices + 1e-12))


def _realised_vol(returns: np.ndarray, window: int) -> np.ndarray:
    """Rolling std dev of log returns."""
    out = np.full(len(returns), np.nan)
    for i in range(window - 1, len(returns)):
        out[i] = returns[i - window + 1 : i + 1].std(ddof=1)
    return out


def _momentum(prices: np.ndarray, window: int) -> np.ndarray:
    """Rate of change momentum."""
    out = np.full(len(prices), np.nan)
    for i in range(window, len(prices)):
        out[i] = (prices[i] - prices[i - window]) / (prices[i - window] + 1e-12)
    return out


def _trend_strength(returns: np.ndarray, window: int) -> np.ndarray:
    """Signed sum of returns over window — proxy for ADX direction."""
    out = np.full(len(returns), np.nan)
    for i in range(window - 1, len(returns)):
        out[i] = returns[i - window + 1 : i + 1].sum()
    return out


def build_feature_matrix(
    prices: np.ndarray,
    cfg: RegimeConfig,
) -> np.ndarray:
    """
    Returns (T, 4) feature matrix:
      col0: z-scored log return
      col1: z-scored realised vol
      col2: z-scored momentum
      col3: z-scored trend strength
    Rows with NaN are dropped; returned alongside valid mask.
    """
    if len(prices) < cfg.min_obs:
        raise ValueError(f"Need >= {cfg.min_obs} price bars, got {len(prices)}")

    rets   = _returns(prices)                              # len T-1
    vol    = _realised_vol(rets, cfg.vol_lookback)         # len T-1
    mom    = _momentum(prices, cfg.momentum_lookback)[1:]  # align to T-1
    trend  = _trend_strength(rets, cfg.zscore_window)      # len T-1

    mat = np.column_stack([rets, vol, mom, trend])         # (T-1, 4)
    valid = ~np.isnan(mat).any(axis=1)
    mat = mat[valid]

    # z-score each feature independently
    mat = zscore(mat, axis=0)
    return mat, valid


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
class RegimeClassifierV2:
    """
    Hidden Markov Model regime classifier.

    Usage:
        clf = RegimeClassifierV2()
        clf.fit(prices)                   # numpy array of close prices
        regimes = clf.predict(prices)     # list[Regime]
        current = clf.current_regime(prices)
    """

    def __init__(self, cfg: Optional[RegimeConfig] = None) -> None:
        self.cfg    = cfg or RegimeConfig()
        self._model: Optional[hmm.GaussianHMM] = None
        self._fitted = False
        self._label_map: dict[int, Regime] = {}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(self, prices: np.ndarray) -> "RegimeClassifierV2":
        """Fit the HMM on historical price data."""
        prices = np.asarray(prices, dtype=float)
        features, _ = build_feature_matrix(prices, self.cfg)

        model = hmm.GaussianHMM(
            n_components    = self.cfg.n_states,
            covariance_type = self.cfg.covariance_type,
            n_iter          = self.cfg.n_iter,
            tol             = self.cfg.tol,
            random_state    = self.cfg.random_state,
            verbose         = False,
        )
        model.fit(features)
        self._model   = model
        self._fitted  = True
        self._label_map = self._build_label_map(features)
        log.info("RegimeClassifierV2 fitted | states=%d | score=%.4f",
                 self.cfg.n_states, model.score(features))
        return self

    # ------------------------------------------------------------------
    # Label mapping — rank hidden states by mean return + vol signature
    # ------------------------------------------------------------------
    def _build_label_map(self, features: np.ndarray) -> dict[int, Regime]:
        """
        After fitting, assign each hidden state a Regime label:
          - Highest mean return    → BULL_TREND
          - Lowest mean return     → BEAR_TREND
          - Highest mean vol       → HIGH_VOL_BREAKOUT
          - Remaining              → RANGING
        """
        hidden = self._model.predict(features)
        state_stats = {}
        for s in range(self.cfg.n_states):
            mask = hidden == s
            if mask.sum() == 0:
                state_stats[s] = {"mean_ret": 0.0, "mean_vol": 0.0}
                continue
            state_stats[s] = {
                "mean_ret": features[mask, 0].mean(),
                "mean_vol": features[mask, 1].mean(),
            }

        ranked_by_ret = sorted(state_stats, key=lambda s: state_stats[s]["mean_ret"])
        ranked_by_vol = sorted(state_stats, key=lambda s: state_stats[s]["mean_vol"])

        label_map: dict[int, Regime] = {}
        label_map[ranked_by_ret[-1]] = Regime.BULL_TREND
        label_map[ranked_by_ret[0]]  = Regime.BEAR_TREND

        # highest vol that isn’t already labelled
        for s in reversed(ranked_by_vol):
            if s not in label_map:
                label_map[s] = Regime.HIGH_VOL_BREAKOUT
                break

        # remainder → RANGING
        for s in range(self.cfg.n_states):
            if s not in label_map:
                label_map[s] = Regime.RANGING

        log.debug("Label map: %s", label_map)
        return label_map

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, prices: np.ndarray) -> list[Regime]:
        """Return a Regime label per valid bar."""
        self._assert_fitted()
        prices   = np.asarray(prices, dtype=float)
        features, valid = build_feature_matrix(prices, self.cfg)
        hidden   = self._model.predict(features)
        return [self._label_map[int(h)] for h in hidden]

    def predict_proba(self, prices: np.ndarray) -> np.ndarray:
        """Return posterior state probabilities (T, n_states)."""
        self._assert_fitted()
        prices   = np.asarray(prices, dtype=float)
        features, _ = build_feature_matrix(prices, self.cfg)
        return self._model.predict_proba(features)

    def current_regime(self, prices: np.ndarray) -> Regime:
        """Return the regime label for the latest bar."""
        return self.predict(prices)[-1]

    def current_proba(self, prices: np.ndarray) -> dict[str, float]:
        """Return {regime_name: probability} for the latest bar."""
        proba  = self.predict_proba(prices)[-1]  # shape (n_states,)
        result = {}
        for state_id, prob in enumerate(proba):
            regime = self._label_map.get(state_id, Regime.RANGING)
            result[regime.value] = float(round(prob, 6))
        return result

    # ------------------------------------------------------------------
    # Transition matrix
    # ------------------------------------------------------------------
    def transition_matrix(self) -> dict:
        """Return the HMM transition matrix as a labelled dict-of-dicts."""
        self._assert_fitted()
        trans = self._model.transmat_
        labels = [self._label_map[i].value for i in range(self.cfg.n_states)]
        return {
            labels[i]: {labels[j]: round(float(trans[i, j]), 6)
                        for j in range(self.cfg.n_states)}
            for i in range(self.cfg.n_states)
        }

    # ------------------------------------------------------------------
    # Model score
    # ------------------------------------------------------------------
    def log_likelihood(self, prices: np.ndarray) -> float:
        """Log-likelihood of the price series under the fitted model."""
        self._assert_fitted()
        prices   = np.asarray(prices, dtype=float)
        features, _ = build_feature_matrix(prices, self.cfg)
        return float(self._model.score(features))

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------
    def _assert_fitted(self) -> None:
        if not self._fitted or self._model is None:
            raise RuntimeError("Call .fit(prices) before predicting.")
