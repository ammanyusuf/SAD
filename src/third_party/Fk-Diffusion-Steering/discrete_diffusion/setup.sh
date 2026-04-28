#!/bin/sh
set -ex

git submodule update --init
## or manually clone our mdlm fork:
# git clone git@github.com:zacharyhorvitz/mdlm.git


pip install -r requirements.txt 
pip install flash-attn

# to use infinigram, you will need to download an index
# For example: `aws s3 cp --no-sign-request --recursive s3://infini-gram-lite/index/v4_dolmasample_olmo <LOCAL_INDEX_PATH>`
# and set the INFINIGRAM_CACHE_DIR environment variable to the path of the local index

# Causal-Conv1d is no longer required with mdlm fork that does not import mamba
#pip install -e git+ssh://git@github.com/Dao-AILab/causal-conv1d.git@82867a9d2e6907cc0f637ac6aff318f696838548#egg=causal_conv1d
