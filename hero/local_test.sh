#!/bin/bash

# Example command for local run.
export JAX_DISABLE_JIT=True; EXP=local_test_1; rm -rf /tmp/${EXP}; python main.py --experiment_config TransformerLMTest --experiment_dir /tmp/${EXP} --verbosity=-1

# Check training curves.
tensorboard --logdir /tmp/${EXP}
