#!/bin/bash
set -e
if [ -d venv ]; then
	echo "ERROR: venv directory already exists, not doing anything."
	exit 1
fi

python3 -m virtualenv venv
source venv/bin/activate

pip install pip --upgrade

pip install torch==1.9

pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu102_pyt190/download.html

pip install -r requirements_cuda.txt
