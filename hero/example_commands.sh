#!/bin/bash

# Example command for installing dependencies.
# JAX installation is environment-specific (CPU, GPU, TPU). Check the official JAX installation guide at https://docs.jax.dev/en/latest/installation.html.
# Examples:
# CPU: `pip install -U jax`
# GPU: `pip install -U "jax[cuda12]"` (Replace `cuda12`; ensure CUDA/cuDNN/drivers are correct)
# TPU: `pip install -U "jax[tpu]"`
pip install -r requirements.txt

# Example command for local run. 
# Add "export JAX_DISABLE_JIT=True;" to disable `jit` for easier debugging.
# Change "TransformerLMTest" to other experiment config names in `config_lib.py` to run other experiments.
EXP=local_test_1; rm -rf /tmp/${EXP}; python main.py --experiment_config TransformerLMTest --experiment_dir /tmp/${EXP} --verbosity=-1

# Example command for checking learning curves with tensorboard.
tensorboard --logdir /tmp/${EXP}
