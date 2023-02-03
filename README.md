<div align="center">

# AICS im2im

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>
<p align="center">
  <img src="docs/overview.png" width="100%" title="project overview">
</p>

## Description
In an effort to spend more work on methods development, simplify maintenance, and acreate a unified framework for all of AICS's image-to-image deep learning tools, we have created `aics_im2im`. We base it on the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) model.

### Supported im2im model types
1. Segmentation
2. Transfer Function
3. Labelfree
4. EmbedSeg
    - Our implementation of EmbedSeg is modified from the [original implementation](https://github.com/juglab/EmbedSeg) and the [MMV-Lab's implementation](https://github.com/MMV-Lab/mmv_im2im). 
    - [Docs](docs/embedseg.md)



## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/AllenCellModeling/aics-im2im
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python aics_im2im/train.py trainer=cpu

# train on GPU
python aics_im2im/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python aics_im2im/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python aics_im2im/train.py trainer.max_epochs=20 datamodule.batch_size=64
```




