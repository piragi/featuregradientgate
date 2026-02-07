"""Compatibility wrapper — canonical source is gradcamfaith.experiments.sweep"""
from gradcamfaith.experiments.sweep import *  # noqa: F401,F403
from gradcamfaith.experiments.sweep import run_single_experiment, run_parameter_sweep, main  # noqa: F401


if __name__ == "__main__":
    main()
    # run_best_performers(subset_size=500)
