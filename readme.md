## Towards Universal Instance Shadow Detection based on Pairwise Grouping with Contrastive Morphological Alignment

This repository contains the official implementation for the paper published in the *IEEE Transactions on Circuits and Systems for Video Technology, 2025*.


**[Towards Universal Instance Shadow Detection based on Pairwise Grouping with Contrastive Morphological Alignment](https://ieeexplore.ieee.org/document/11181182)**


**Authors: Haopeng Fang, Fei Liu, Wenfeng Han, and He Tang**


-----

### 📦 Setup and Installation

#### Requirements

First, install the basic requirements:

```bash
pip install -r requirements.txt
```

#### Example `conda` Environment Setup

We recommend setting up a dedicated environment using `conda`:

```bash
conda create --name shadow python=3.8 -y
conda activate shadow

# Install PyTorch with CUDA 11.3 support
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install OpenCV
pip install -U opencv-python

# Install Detectron2 (under your working directory)
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install Panoptic API and Cityscapes Scripts
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git

# Install remaining requirements
pip install -r requirements.txt

# Compile CUDA kernel for MSDeformAttn
cd mask2former/modeling/pixel_decoder/ops
sh make.sh

# Install PythonAPI (if required by the project structure)
cd ../../../../
cd PythonAPI 
python setup.py install
```

#### Compiling CUDA kernel for MSDeformAttn

The MSDeformAttn CUDA kernel is required for this model. Ensure your `CUDA_HOME` environment variable is defined and points to your CUDA toolkit installation directory.

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

**Building on a system without a GPU:**

If you are building on a machine that has the necessary drivers but no GPU device, you can force the compilation for a specific CUDA architecture (e.g., 8.0 for Ampere/RTX 30 series):

```bash
TORCH_CUDA_ARCH_LIST='8.0' FORCE_CUDA=1 python setup.py build install
```

-----

### 📦 Dataset Preparation

Download the **SOBAv2 Dataset** from the following source:

```bash
You can download the SOBAv2 dataset from https://github.com/stevewongv/SSIS and unzip it to the "dataset" path.
```

Ensure the dataset is unzipped into a directory named `dataset` (or update the configuration files to point to the correct path).

-----

### 📦 Training and Evaluation

#### Training

To train the model using the provided configuration on a single GPU:

```bash
python train_net.py --config-file configs/instance-segmentation/maskformer2_R101_bs16_50ep.yaml --num-gpus 1
```

#### Evaluation

To evaluate the model's performance on a single GPU, you need to specify the path to your trained model weights:

```bash
python train_net.py --config-file configs/instance-segmentation/maskformer2_R101_bs16_50ep.yaml --num-gpus 1 --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```



### Acknowledgments

This project is built upon the foundational work of several outstanding open-source projects. We sincerely thank the authors and maintainers of the following repositories for their contributions to the research community, which greatly facilitated our work:

* **[Mask2Former](https://github.com/facebookresearch/Mask2Former)**
* **[MaskFormer](https://github.com/facebookresearch/MaskFormer)**
* **[SSIS & SSISv2](https://github.com/stevewongv/SSIS)**
* **[LDF](https://github.com/weijun-arc/LDF)**

---

If you find this work useful, please consider citing our paper:

```bibtex
@article{fang2025towards,
  title={Towards Universal Instance Shadow Detection based on Pairwise Grouping with Contrastive Morphological Alignment},
  author={Fang, Haopeng and Liu, Fei and Han, Wenfeng and Tang, He},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  publisher={IEEE}
}
```