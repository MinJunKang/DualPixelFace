# DualPixelFace
Official pytorch implementation of ["Facial Depth and Normal Estimation using Single Dual-Pixel Camera"](https://arxiv.org/abs/2111.12928) (ECCV 2022)

[Minjun Kang](http://rcv.kaist.ac.kr/), [Jaesung Choe](https://sites.google.com/view/jaesungchoe), [Hyowon Ha](https://sites.google.com/site/hyowoncv/), [Hae-Gon Jeon](https://sites.google.com/site/hgjeoncv/home), [Sunghoon Im](https://cvlab.dgist.ac.kr/), [In So Kweon](http://rcv.kaist.ac.kr/), and [KuK-Jin Yoon](http://vi.kaist.ac.kr/)

[pdf] [project] [bibtex]

<img width="80%" src="https://github.com/MinJunKang/DualPixelFace/tree/main/asset/teaser.pdf"/>

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

## Supporting Dataset

### Depth Benchmark
(1) Google dual pixel depth benchmark. 

(See https://github.com/google-research/google-research/tree/master/dual_pixels for detail information).

(2) Our facial dataset benchmark. (comming soon, available by contacting us.)

If you use these datasets, please cite their papers.

## Supporting Model

(1) PSMNet      [[Paper](https://arxiv.org/abs/1803.08669)]       [[Code](https://github.com/JiaRenChang/PSMNet)]

(2) DPNet       [[Paper](https://arxiv.org/abs/1904.05822)]      [[Project](https://github.com/google-research/google-research/tree/master/dual_pixels)]

(3) StereoNet       [[Paper](https://arxiv.org/abs/1807.08865)]       [[Code](https://github.com/meteorshowers/X-StereoLab)]

(4) NNet       [[Paper](https://arxiv.org/abs/1911.10444)]       [[Code](https://github.com/udaykusupati/Normal-Assisted-Stereo)]

(5) BTS       [[Paper](https://arxiv.org/abs/1907.10326)]       [[Code](https://github.com/cleinc/bts)]

(6) StereoDPNet      (Ours)

If you use these models, please cite their papers.

## How to run?

First, get the dataset! and put your dataset's location at config.py 's path:datapath !!

**Example (Our facial dataset benchmark):** 

- /home/miru/rcv_face/HighRes
  - 2020-1-16_group3
  - ...
  - 2020-2-19_group25
  - test.txt
  - train.txt

Set path:datapath = "/home/miru/rcv_face/HighRes"

### Training & Validation

<pre>
<code>
CUDA_VISIBLE_DEVICES=[gpu idx] python main.py --config train_faceDP --workspace [Workspace Name]
</code>
</pre>

The result will be automatically saved in ./output/[model to use]/[Workspace name]/*.

### Testing

If you want to use your own pretrained weight, please run like this.

<pre>
<code>
CUDA_VISIBLE_DEVICES=[gpu idx] python main.py --config train_faceDP --workspace [Workspace Name] --loadmodel [relative/absolute path to checkpoint]
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
