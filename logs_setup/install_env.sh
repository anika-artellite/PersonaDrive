#!/bin/bash
source /usr/share/Modules/init/bash
module load miniconda3/25.11.1
export PIP_CACHE_DIR=/dfs6b/pub/asaha6/cache/pip
export CONDA_PKGS_DIRS=/dfs6b/pub/asaha6/cache/conda_pkgs
export CONDA_DEFAULT_CHANNELS=https://conda.anaconda.org/conda-forge
cd /dfs6b/pub/asaha6/agent/PersonaDrive
bash environment/setup_env.sh -p /dfs6b/pub/asaha6/envs/personadrive
echo "ENV_EXIT=$?"
