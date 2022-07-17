# DualPixelFace
Official pytorch implementation of 

["Facial Depth and Normal Estimation using Single Dual-Pixel Camera"](https://arxiv.org/abs/2111.12928) (ECCV 2022)

[Minjun Kang](http://rcv.kaist.ac.kr/), [Jaesung Choe](https://sites.google.com/view/jaesungchoe), [Hyowon Ha](https://sites.google.com/site/hyowoncv/), [Hae-Gon Jeon](https://sites.google.com/site/hgjeoncv/team), [Sunghoon Im](https://sunghoonim.github.io/) and [In So Kweon](http://rcv.kaist.ac.kr/)

[pdf] [project] [bibtex]

## Project Description
This project aims to provide environment for all the developlers/researchers working with the dual pixel sensor.

Please update any issues and revised code using the function of pull requests.

This project provides benchmark dataset and baseline code of "Facial Depth and Normal Estimation using Single Dual-Pixel Camera".

You can also see the recent papers of Dual-Pixel in this [page](https://github.com/MinJunKang/DualPixelFace/blob/main/Reference.md).

## Environments

Tested on Ubuntu 18.04 CUDA-10.1 (10.2) with Pytorch==1.5.0, Torchvision==0.6.0 (python version 3.6)

## Supported Dataset

### Depth Benchmark
(1) Google dual pixel depth benchmark. 

(See https://github.com/google-research/google-research/tree/master/dual_pixels for detail information).

(2) Our facial dataset benchmark. (comming soon, available by contacting us.)

### Deblurring Benchmark
(1) York University dual pixel defocus deblurring benchmark.

(See https://github.com/Abdullah-Abuolaim/defocus-deblurring-dual-pixel for detail information)

## Supported Model

(1) PSMNet      [[Paper](https://arxiv.org/abs/1803.08669)]       [[Code](https://github.com/JiaRenChang/PSMNet)]

(2) GWCNet      [[Paper](https://arxiv.org/abs/1903.04025)]      [[Code](https://github.com/xy-guo/GwcNet)]

(3) DPNet       [[Paper](https://arxiv.org/abs/1904.05822)]      [Code]

(4) SubNormal      [[Paper]()]

## Environment Setting

Your main workspace will be "DPStudioLighten".

<pre>
<code>
conda create -n dpstudio python=3.6
conda activate dpstudio
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 (10.2) -c pytorch
cd DPStudioLighten
sh installer.sh
</code>
</pre>

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
cd DPStudioLighten
CUDA_VISIBLE_DEVICES=[gpu idx] python Main.py --model [model to use] --workspace [Workspace name] --config config --dataset rcv_face --ngpu [#gpu to use] --batchsize [batch size]
</code>
</pre>

The result will be automatically saved in ./output/[model to use]/[Workspace name]/*.

If you want to use your own pretrained weight, please run like this.

<pre>
<code>
cd DPStudioLighten
CUDA_VISIBLE_DEVICES=[gpu idx] python Main.py --model pairnet --workspace [Workspace name] --config config --dataset rcv_face --ngpu [#gpu to use] --batchsize [batch size] --loadmodel [Your relative/absolute path to checkpoint]
</code>
</pre>

### Testing & Demo

<pre>
<code>
cd DPStudioLighten
CUDA_VISIBLE_DEVICES=[gpu idx] python Main.py --model pairnet --workspace [Workspace name] --config config --dataset rcv_face --ngpu [#gpu to use] --loadmodel [Your relative/absolute path to checkpoint] --mode test
</code>
</pre>

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
