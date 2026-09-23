"""
Backward-compatible wrapper. The Genetic Algorithm now lives in
src/optimizers.py alongside PSO and GWO (see run_ablation_study.py for the
optimizer comparison this enables). This module re-exports it under its
original name so any existing scripts/imports keep working.
"""

from src.optimizers import ga_search as run_genetic_search  # noqa: F401
