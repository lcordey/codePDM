#!/bin/bash
set -e
if [ -d venv ]; then
	echo "ERROR: venv directory already exists, not doing anything."
	exit 1
fi

python3 -m virtualenv venv
source venv/bin/activate

pip install pip --upgrade

pip install torch==1.10

curl -LO https://github.com/NVIDIA/cub/archive/1.10.0.tar.gz
tar xzf 1.10.0.tar.gz && rm 1.10.0.tar.gz
export CUB_HOME=`pwd`/cub-1.10.0
pip install -r requirements.txt
