# QTIP — Quantitative Trading Intelligence Platform

> **Owner**: Richard Patterson © 2026 — De-ASI-INTERFACE
> **Version**: 2.0.0 | **Status**: Production-Hardened & Fully Tested

---

## Regime Classifier v2

Hidden Markov Model (HMM)-based market regime classifier that labels price action into four states:

| Regime | Description |
|---|---|
| `BULL_TREND` | Positive drift, low-moderate vol |
| `BEAR_TREND` | Negative drift, low-moderate vol |
| `RANGING` | Near-zero drift, low vol |
| `HIGH_VOL_BREAKOUT` | Any drift, high vol |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/De-ASI-INTERFACE/qtip.git
cd qtip

# 2. Install in editable mode
pip install -e .

# 3. Install dev deps
pip install -e ".[dev]"

# 4. Run full test suite with 90% coverage gate
pytest
```

---

## Usage

```python
import numpy as np
from qtip import RegimeClassifierV2, Regime

# Fit on historical close prices
prices = np.array([...])  # numpy array of close prices
clf = RegimeClassifierV2()
clf.fit(prices)

# Predict regime per bar
regimes = clf.predict(prices)

# Current regime
current = clf.current_regime(prices)
print(current)  # Regime.BULL_TREND

# Posterior probabilities for latest bar
probs = clf.current_proba(prices)
# {'BULL_TREND': 0.821, 'BEAR_TREND': 0.04, 'RANGING': 0.11, 'HIGH_VOL_BREAKOUT': 0.029}

# HMM transition matrix
tm = clf.transition_matrix()

# Log-likelihood score
ll = clf.log_likelihood(prices)
```

---

## Architecture

```
qtip/
├── __init__.py
└── regime_classifier_v2.py
    ├── Regime (enum)
    ├── RegimeConfig (dataclass)
    ├── build_feature_matrix()
    └── RegimeClassifierV2
        ├── .fit(prices)
        ├── .predict(prices)
        ├── .predict_proba(prices)
        ├── .current_regime(prices)
        ├── .current_proba(prices)
        ├── .transition_matrix()
        └── .log_likelihood(prices)

tests/
└── test_regime_classifier_v2.py   # 35 tests, 90%+ coverage gate

.github/workflows/
└── regime-ci.yml                  # lint → bandit → pytest

pyproject.toml                     # editable install + pytest config
```

---

## Features

- 4-state Gaussian HMM with configurable covariance (`full` / `diag` / `tied` / `spherical`)
- Automatic regime labelling by mean return + vol signature — no manual state assignment
- Z-scored feature matrix: log returns, realised vol, momentum (ROC), trend strength
- Posterior state probabilities per bar
- Full transition matrix with regime labels
- Log-likelihood scoring for model comparison
- 90%+ branch coverage enforced in CI

---

## Dependencies

| Package | Min Version | Purpose |
|---|---|---|
| `hmmlearn` | 0.3.0 | Gaussian HMM engine |
| `numpy` | 1.26.0 | Feature computation |
| `scipy` | 1.12.0 | Z-score normalisation |
| `pytest` | 8.0.0 | Test runner (dev) |
| `pytest-cov` | 5.0.0 | Coverage gate (dev) |

---

## Roadmap

- [x] Regime Classifier v2 (HMM-based state transitions)
- [ ] Full signal ingestion pipeline (Binance, Solana, on-chain)
- [ ] Amplitude scorer with backtested weight calibration
- [ ] REST API output interface for downstream bot consumption
- [ ] Docker Compose deployment with Prometheus integration

---

## License

Proprietary — Richard Patterson © 2026. All rights reserved.
