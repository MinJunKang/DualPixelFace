# Facial Depth and Normal Estimation using Single Dual-Pixel Camera
Official pytorch implementation of ["Facial Depth and Normal Estimation using Single Dual-Pixel Camera"](https://arxiv.org/abs/2111.12928) (ECCV 2022)

[Minjun Kang](http://rcv.kaist.ac.kr/), [Jaesung Choe](https://sites.google.com/view/jaesungchoe), [Hyowon Ha](https://sites.google.com/site/hyowoncv/), [Hae-Gon Jeon](https://sites.google.com/site/hgjeoncv/home), [Sunghoon Im](https://cvlab.dgist.ac.kr/), [In So Kweon](http://rcv.kaist.ac.kr/), and [KuK-Jin Yoon](http://vi.kaist.ac.kr/)

[Paper] [Project] [YouTube] [PPT]

<img src="https://github.com/MinJunKang/DualPixelFace/blob/main/asset/teaser.png" alt="drawing" width = "890">

## Project Description
This project aims to provide face related dual-pixel benchmark for all the developlers/researchers working with the dual pixel sensor.

This project provides benchmark dataset and baseline code of "Facial Depth and Normal Estimation using Single Dual-Pixel Camera".

You can also see the recent papers related to Dual-Pixel in this [page](https://github.com/MinJunKang/DualPixelFace/blob/main/Reference.md).

## Environment Setting

**Conda Environment**
: Ubuntu 18.04 CUDA-10.1 (10.2) with Pytorch==1.5.0, Torchvision==0.6.0 (python version 3.6).
<pre>
<code>
# Create Environment
conda create -n dpface python=3.6
conda activate dpface

# Install pytorch, torchvision, cudatoolkit
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 (10.2) -c pytorch

# Install package and cuda build
sh ./installer.sh
</code>
</pre>

**Docker Environment**
: Ubuntu 18.04 CUDA-10.2 with Pytorch==1.6.0, Torchvision==0.7.0 (python version 3.7).
<pre>
<code>
# Pull docker image
docker push jack4852/eccv22_facialdocker:latest

# create container and include dataset's path
docker run -it -d --gpus all --name dpface --shm-size 64G --mount type=bind,source=[Dataset Path],target=/ndata jack4852/eccv22_facialdocker:latest

# start container
docker start dpface

# attach container
docker attach dpface

# pull the code from github
git init
git pull https://github.com/MinJunKang/DualPixelFace

# Install package and cuda build
sh ./installer.sh

</code>
</pre>

Our code is based on [PytorchLightning](https://www.pytorchlightning.ai/).

## Supporting Datasets

### (1) Facial dual-pixel benchmark.

(Since dataset is huge (~600G), we are now providing download link for the researchers who request the dataset.)

**How to get dataset?**

1. Download, read [LICENSE AGREEMENT](https://drive.google.com/file/d/1HaYd8fqxoeAAtcCZzkAYQe9KwgAKwpgA/view?usp=sharing), and confirm that all the terms are agreed. Then scan the signed [LICENSE AGREEMENT](https://drive.google.com/file/d/1HaYd8fqxoeAAtcCZzkAYQe9KwgAKwpgA/view?usp=sharing). (Electronic signature is allowed.)
2. Send an email to kmmj2005@gmail.com with your signed agreement.

**Directory Structure of our Face Dataset**
<pre>
<code>
- Parent Directory
  - 2020-1-15_group2
  - 2020-1-16_group3
    - NORMAL                : surface normal (*.npy)
    - MASK                  : mask obtained from Structured Light (*.npy)
    - JSON                  : including path, calibration info (*.json)
    - IMG                   : IMG of LEFT, RIGHT, LEFT + RIGHT (*.JPG)
    - DEPTH                 : metric-scale depth [mm] (*.npy)
    - CALIBRATION
      - pose.npy            : camera extrinsics (8 cameras)
      - Metadata.npy        : focal length [mm], focal distance [mm], Fnumber, pixel size [um]
      - light.npy           : light direction of 6 different light conditions
      - intrinsic.npy       : intrinsic matrix (8 cameras)
      - Disp2Depth.npy      : currently not used
    - ALBEDO                : albedo map (*.npy)
  - ...
  - 2020-2-19_group25
  - test.txt                : list of directories for test set
  - train.txt               : list of directories for training set
</code>
</pre>

### (2) Google dual-pixel depth benchmark.

(See https://github.com/google-research/google-research/tree/master/dual_pixels for detail information).

If you use these datasets, please cite their papers.

## Supporting Models

(1) PSMNet      [[Paper](https://arxiv.org/abs/1803.08669)]       [[Code](https://github.com/JiaRenChang/PSMNet)]       [pretrained weight]

(2) DPNet       [[Paper](https://arxiv.org/abs/1904.05822)]      [[Project](https://github.com/google-research/google-research/tree/master/dual_pixels)]       [pretrained weight]

(3) StereoNet       [[Paper](https://arxiv.org/abs/1807.08865)]       [[Code](https://github.com/meteorshowers/X-StereoLab)]       [pretrained weight]

(4) NNet       [[Paper](https://arxiv.org/abs/1911.10444)]       [[Code](https://github.com/udaykusupati/Normal-Assisted-Stereo)]       [[pretrained weight](https://drive.google.com/file/d/1R6SuGC1Z50tx4e1DlpfaVgcSFFM3XxJp/view?usp=sharing)]

(5) BTS       [[Paper](https://arxiv.org/abs/1907.10326)]       [[Code](https://github.com/cleinc/bts)]       [pretrained weight]

(6) StereoDPNet      (Ours)       [[pretrained weight](https://drive.google.com/file/d/1nf3R1y4Op8jeexQ9h8wgllkgGJIt-MNX/view?usp=sharing)]

If you use these models, please cite their papers.

## Instructions for Code

### Code Structure (Simple rule for name)

*config_/*.json : set dataset, model, and augmentations to apply by assigning configuration file name.*
*src/model/[model_name] : If you want to add your own model, main class name should be the same as upper case of "model_name".*
*src/dataloader/[dataset_name] : If you want to add your own dataset, main class name should be the "[dataset_name]Loader".*

### Training & Validation

<pre>
<code>
CUDA_VISIBLE_DEVICES=[gpu idx] python main.py --config train_faceDP --workspace [Workspace Name]
</code>
</pre>

The result will be automatically saved in ./workspace/[model name]/[Workspace Name]/*.
You can change the model to run by changing "model_name" parameter in config_/train_faceDP.json. (must be the same as the model's name of src/model)

### Testing

If you want to use your own pretrained weight, please run like this.

<pre>
<code>
CUDA_VISIBLE_DEVICES=[gpu idx] python main.py --config eval_faceDP --workspace [Workspace Name] --load_model [relative/absolute path to checkpoint]
</code>
</pre>

### Demo

Will be updated soon!

### References
<pre>
<code>
@article{kang2021facial,
  title={Facial Depth and Normal Estimation using Single Dual-Pixel Camera},
  author={Kang, Minjun and Choe, Jaesung and Ha, Hyowon and Jeon, Hae-Gon and Im, Sunghoon and Kweon, In So},
  journal={arXiv preprint arXiv:2111.12928},
  year={2021}
}
</code>
</pre>
