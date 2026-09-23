"""
Shared hyperparameter search-space definition used by every optimizer
(Genetic Algorithm, Particle Swarm Optimization, Grey Wolf Optimizer) in
src/optimizers.py.

For a fair comparison between optimization algorithms -- the paper's own
stated future direction ("exploration of a wider range of optimization
techniques beyond Genetic Algorithms... including Grey Wolf, Particle
Swarm, jSO") -- all three algorithms must search the *same* encoded
space, using the *same* decode function and the *same* fitness function.
This module is that shared contract.

Encoding: each candidate hyperparameter set is a 4-dimensional continuous
vector x = [x0, x1, x2, x3] with:
    x0 in [0, 3] -> rounded to nearest int -> index into DENSE_UNIT_CHOICES
    x1 in DROPOUT_RANGE                    -> dropout_rate directly
    x2 in LR_LOG_RANGE                     -> learning_rate = 10 ** x2
    x3 in [0, 2] -> rounded to nearest int -> index into OPTIMIZER_CHOICES
"""

import numpy as np

DENSE_UNIT_CHOICES = [64, 128, 256, 512]
OPTIMIZER_CHOICES = ["adam", "rmsprop", "sgd"]
LR_LOG_RANGE = (-5.0, -2.0)     # 1e-5 to 1e-2
DROPOUT_RANGE = (0.2, 0.6)

DIM = 4
BOUNDS = np.array([
    [0.0, len(DENSE_UNIT_CHOICES) - 1],
    [DROPOUT_RANGE[0], DROPOUT_RANGE[1]],
    [LR_LOG_RANGE[0], LR_LOG_RANGE[1]],
    [0.0, len(OPTIMIZER_CHOICES) - 1],
])
LOWER = BOUNDS[:, 0]
UPPER = BOUNDS[:, 1]


def random_vector(rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(LOWER, UPPER)


def clip_vector(x: np.ndarray) -> np.ndarray:
    return np.clip(x, LOWER, UPPER)


def decode(x: np.ndarray) -> dict:
    """Decodes a raw 4D vector into a concrete hyperparameter dict."""
    x = clip_vector(x)
    dense_idx = int(round(x[0]))
    opt_idx = int(round(x[3]))
    return {
        "dense_units": int(DENSE_UNIT_CHOICES[dense_idx]),
        "dropout_rate": float(x[1]),
        "learning_rate": float(10 ** x[2]),
        "optimizer_name": str(OPTIMIZER_CHOICES[opt_idx]),
    }


def vector_repr(x: np.ndarray) -> str:
    d = decode(x)
    return (f"dense_units={d['dense_units']}, dropout={d['dropout_rate']:.3f}, "
            f"lr={d['learning_rate']:.2e}, optimizer={d['optimizer_name']}")
