# Sample Selection via Contrastive Fragmentation for Noisy Label Regression
This repository contains the official PyTorch implementation for our ICCV2021 paper.
- [Chris Dongjoo Kim*](https://cdjkim.github.io/), Sangwoo Moon*, Jihwan Moon, [Dongyeon Woo](https://woody0325.github.io/), [Gunhee Kim](https://vision.snu.ac.kr/gunhee/). Sample Selection via Contrastive Fragmentation for Noisy Label Regression. In NeurIPS, 2024 (* equal contribution).

[[Paper Link]](https://arxiv.org/abs/2110.07735)

## System Dependencies
- Python >= 3.9
- CUDA >= 9.0 supported GPU

## Installation
Using virtual env is recommended.
```
# create conda env with python=3.9
conda create -n {ENV_NAME} python=3.9

conda activate {ENV_NAME}

# install required version of torch and torchvision
pip install -f https://download.pytorch.org/whl/torch_stable.html torch==1.13.1+cu116 torchvision==0.14.1+cu116

# install other packages
pip install -r requirements.txt
```


## Preparation
### Log directory set-up
Create `checkpoints` directory for logging.
```
$ ln -s [log directory path] checkpoints
```

### Data set-up:
1. Download `data.zip` from [here](https://drive.google.com/file/d/1srpSDBM30dVmfa80wdIG-8oTpbgM6dba/view?usp=sharing).
2. Decompress into `./data`.


## Run
Specify parameters in `config` yaml, `episodes` yaml files.
```
python main.py --log-dir [log directory path] --config [config file path] --episode [episode file path] --override "|" --random_seed [seed]

```

Example of ConFrag IMDB-clean-bal run
```
python main.py --config=configs/imdb_fragment.yaml --episode=episodes/imdb-split4.yaml --log-dir=checkpoints/imdb/dfragment/imdb_clean_bal --random_seed=[seed]
```


## Citation
The code and dataset are free to use for academic purposes only. If you use any of the material in this repository as part of your work, we ask you to cite:
```
@inproceedings{kim-NeurIPS-2024,
    author    = {Chris Dongjoo Kim and Sangwoo Moon and Jihwan Moon and Dongyeon Woo and Gunhee Kim},
    title     = "{Sample Selection via Contrastive Fragmentation for Noisy Label Regression}"
    booktitle = {NeurIPS},
    year      = 2024
}
```
