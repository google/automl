#!/bin/bash

# Example command for local run. 
# Add "export JAX_DISABLE_JIT=True;" to disable `jit` for easier debugging.
# Change "TransformerLMTest" to other experiment config names in `config_lib.py` to run other experiments.
EXP=local_test_1; rm -rf /tmp/${EXP}; python main.py --experiment_config TransformerLMTest --experiment_dir /tmp/${EXP} --verbosity=-1

# Check training curves.
tensorboard --logdir /tmp/${EXP}
