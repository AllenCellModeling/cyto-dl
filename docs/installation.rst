.. highlight:: shell

============
Installation
============

``
git clone https://github.com/AllenCellModeling/aics-im2im
cd aics-im2im

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

pip install -r requirements/requirements.txt

# [OPTIONAL] install extra dependencies - equivariance related
pip install -r requirements/equiv-requirements.txt

# [OPTIONAL] Useful if pip install -e . doesn't work
pip install --upgrade pip

pip install -e .
``
