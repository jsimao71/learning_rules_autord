"""Editable local learning rule.

This is the ONLY source file the autonomous R&D agent may modify.
The evaluator imports this module through a fixed API.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class RuleConfig:
    lr: float = 0.03
    decay: float = 0.002
    competition: float = 1.0
    homeostasis: float = 0.01
    target_activity: float = 0.10


PARAM_BOUNDS = {
    "lr": (1e-4, 0.2),
    "decay": (0.0, 0.05),
    "competition": (0.1, 8.0),
    "homeostasis": (0.0, 0.2),
    "target_activity": (0.02, 0.5),
}


def init_state(n_pre: int, n_post: int, rng: np.random.Generator):
    """Create strictly layer-local mutable state."""
    return {
        "post_mean": np.zeros(n_post, dtype=float),
    }


def activate(x, w, state, cfg: RuleConfig):
    """Competitive feed-forward activation using only local layer variables."""
    z = x @ w.T
    z = z - np.max(z, axis=-1, keepdims=True)
    p = np.exp(cfg.competition * z)
    p /= np.sum(p, axis=-1, keepdims=True) + 1e-12
    return p


def update(x, y, w, state, cfg: RuleConfig):
    """Baseline competitive Hebbian/Oja-like local update.

    x: [batch, n_pre]
    y: [batch, n_post]
    w: [n_post, n_pre]
    """
    batch = max(1, x.shape[0])
    hebb = (y.T @ x) / batch
    # Local Oja-like stabilizer: active neurons pay a weight-proportional cost.
    post_power = np.mean(y * y, axis=0)[:, None]
    dw = cfg.lr * (hebb - post_power * w - cfg.decay * w)

    state["post_mean"] = 0.99 * state["post_mean"] + 0.01 * np.mean(y, axis=0)
    # Homeostatic gain is implemented as a local row-scale pressure.
    pressure = cfg.homeostasis * (cfg.target_activity - state["post_mean"])[:, None]
    dw += cfg.lr * pressure * w
    return w + dw, state


def complexity_score():
    """Approximate mechanism complexity used only as a small tie-break penalty."""
    return 5.0
