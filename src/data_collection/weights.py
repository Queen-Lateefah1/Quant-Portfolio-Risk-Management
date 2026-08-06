"""
weights.py

Superseded by src/portfolio/weights.py — Phase 3 (Portfolio Construction)
is implemented there, since portfolio weighting is a portfolio-level
concern rather than a data-collection one. This file re-exports the
real implementation so nothing referencing this old path breaks.
"""

from src.portfolio.weights import (  # noqa: F401
    equal_weights,
    market_cap_weights,
    user_defined_weights,
)
