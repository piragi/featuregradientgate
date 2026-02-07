"""Compatibility wrapper — canonical source is gradcamfaith.experiments.sae_train"""
from gradcamfaith.experiments.sae_train import *  # noqa: F401,F403
from gradcamfaith.experiments.sae_train import train_single_config, main, SWEEP_CONFIG  # noqa: F401


if __name__ == "__main__":
    results = main()
