#!/usr/bin/env bash
PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

eval "$(conda shell.bash hook)"
# conda activate python3.6
DRIVER_SCRIPT='robust-lsm-trees.py'
python ${PARENT_PATH}'/../'${DRIVER_SCRIPT}
conda deactivate
