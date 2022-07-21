# Face Reconstruction from Dual-Pixel Camera

This is an official implementation of the paper,
> [Facial Depth and Normal Estimation using Single Dual-Pixel Camera](https://arxiv.org/abs/2111.12928)<br/>
> [Minjun Kang](http://rcv.kaist.ac.kr/), [Jaesung Choe](https://sites.google.com/view/jaesungchoe), [Hyowon Ha](https://sites.google.com/site/hyowoncv/), [Hae-Gon Jeon](https://sites.google.com/site/hgjeoncv/home), [Sunghoon Im](https://cvlab.dgist.ac.kr/), [In So Kweon](http://rcv.kaist.ac.kr/), and [KuK-Jin Yoon](http://vi.kaist.ac.kr/)<br/>
> European Conference on Computer Vision (ECCV), Tel Aviv, Israel, 2022<br/>
> [Paper](https://arxiv.org/abs/2111.12928) [Project] [YouTube] [PPT] [Dataset](##facial-Dual-Pixel-Benchmark)


## Project Description
- Provide face related dual-pixel benchmark for all the developers/researchers working with the dual pixel sensor.
- Release new benchmark dataset and baseline code.
- Summarize awesome Dual-Pixel papers [Page](https://github.com/MinJunKang/DualPixelFace/blob/main/Reference.md).
<img src="https://github.com/MinJunKang/DualPixelFace/blob/main/asset/teaser.png" alt="drawing" width = "890">


## Environment Setting
- **Conda environment**
: Ubuntu 18.04 CUDA-10.1 (10.2) with Pytorch==1.5.0, Torchvision==0.6.0 (python 3.6).<br/>
```
# Create Environment
conda create -n dpface python=3.6
conda activate dpface

# Install pytorch, torchvision, cudatoolkit
conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.1 (10.2) -c pytorch

# Install package and cuda build
sh ./installer.sh
```

- **Docker environment**
: Ubuntu 18.04 CUDA-10.2 with Pytorch==1.6.0, Torchvision==0.7.0 (python 3.7).
```
# Pull docker image
docker pull jack4852/eccv22_facialdocker:latest

# create container and include dataset's path
docker run -it -d --gpus all --name dpface --shm-size 64G --mount type=bind,source=[Local Dataset Path],target=[Docker Dataset Path] jack4852/eccv22_facialdocker:latest

# start container
docker start dpface

# attach container
docker attach dpface

# pull the code from github
git init
git pull https://github.com/MinJunKang/DualPixelFace

# Install package and cuda build
sh ./installer.sh
```

## Facial Dual-Pixel Benchmark

(Since dataset is huge (~600G), we are now providing download link for the researchers who request the dataset.)

- **How to get dataset?**

1. Download, read [LICENSE AGREEMENT](https://drive.google.com/file/d/1HaYd8fqxoeAAtcCZzkAYQe9KwgAKwpgA/view?usp=sharing), and confirm that all the terms are agreed.
2. Then scan the signed [LICENSE AGREEMENT](https://drive.google.com/file/d/1HaYd8fqxoeAAtcCZzkAYQe9KwgAKwpgA/view?usp=sharing). (Digital signature is allowed.)
3. Send an email to kmmj2005@gmail.com with your signed agreement.

- **Directory structure of our dataset**
```
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
```

## Supporting Models

(1) PSMNet      [[Paper](https://arxiv.org/abs/1803.08669)]       [[Code](https://github.com/MinJunKang/DualPixelFace/tree/main/src/model/psmnet)]       [pretrained weight]

(2) DPNet       [[Paper](https://arxiv.org/abs/1904.05822)]      [[Code](https://github.com/MinJunKang/DualPixelFace/tree/main/src/model/dpnet)]       [pretrained weight]

(3) StereoNet       [[Paper](https://arxiv.org/abs/1807.08865)]       [[Code](https://github.com/MinJunKang/DualPixelFace/tree/main/src/model/stereonet)]       [pretrained weight]

(4) NNet       [[Paper](https://arxiv.org/abs/1911.10444)]       [[Code](https://github.com/MinJunKang/DualPixelFace/tree/main/src/model/nnet)]       [[pretrained weight](https://drive.google.com/file/d/1R6SuGC1Z50tx4e1DlpfaVgcSFFM3XxJp/view?usp=sharing)]

(5) BTS       [[Paper](https://arxiv.org/abs/1907.10326)]       [[Code](https://github.com/MinJunKang/DualPixelFace/tree/main/src/model/bts)]       [pretrained weight]

(6) StereoDPNet      **(Ours)**       [[Code](https://github.com/MinJunKang/DualPixelFace/tree/main/src/model/stereodpnet)]       [[pretrained weight](https://drive.google.com/file/d/1nf3R1y4Op8jeexQ9h8wgllkgGJIt-MNX/view?usp=sharing)]

If you use these models, please cite their papers.

## Instructions for Code

### Code Structure (Simple rule for name)

- config_/[main config].json : set options of dataset, model, and augmentations to use.

- src/model/[model_name] : If you want to add your own model, main class name should be the upper case of "model_name".

  (The model should contain json file that indicates specific parameters of the model.)

- src/dataloader/[dataset_name] : If you want to add your own dataset, main class name should be the "[dataset_name]Loader".

  (The dataset should contain json file that indicates specific parameters of the dataset.)

- You can set the model to run by setting "model_name" parameter in config_/[main config].json. 

  (must be the same as the model_name of src/model)

### Training & Validation
```
CUDA_VISIBLE_DEVICES=[gpu idx] python main.py --config [main config] --workspace [Workspace Name]
```

The results will be automatically saved in ./workspace/[model name]/[Workspace Name].

**Example (1).** Train StereoDPNet with our face dataset 

(results and checkpoints are saved in ./workspace/stereodpnet/base)
```
CUDA_VISIBLE_DEVICES=[gpu idx] python main.py --config train_faceDP --workspace base
```

**Example (2).** Train DPNet with our face dataset 

(results and checkpoints are saved in ./workspace/dpnet/base2)
```
CUDA_VISIBLE_DEVICES=[gpu idx] python main.py --config train_faceDP_dpnet --workspace base2
```

**Example (3).** Resume training of StereoDPNet with our face dataset 

(results and checkpoints are saved in ./workspace/stereodpnet/base2)
```
CUDA_VISIBLE_DEVICES=[gpu idx] python main.py --config train_faceDP --workspace base2 --load_model [path to checkpoint]
```


### Testing
If you want to use your own pretrained weight for test, please run like this.

```
CUDA_VISIBLE_DEVICES=[gpu idx] python main.py --config eval_faceDP --workspace [Workspace Name] --load_model [relative/absolute path to checkpoint]
```

### Demo
Will be updated soon!

## Acknowledgements
This work is in part supported by the Ministry of Trade, Industry and Energy (MOTIE) and Korea Institute for Advancement of Technology (KIAT) through the International Cooperative R\&D program in part (P0019797), `Project for Science and Technology Opens the Future of the Region' program through the INNOPOLIS FOUNDATION funded by Ministry of Science and ICT (Project Number: 2022-DD-UP-0312), and also supported by the Samsung Electronics Co., Ltd (Project Number: G01210570).

[Face-Segmentation-Tool](https://github.com/zllrunning/face-parsing.PyTorch) : We use this repo to get face mask for demo at [here](https://github.com/MinJunKang/DualPixelFace/tree/main/src/module/face_seg).

[3D Deformable Conv](https://github.com/XinyiYing/D3Dnet) : We use this repo to implement ANM module of StereoDPNet at [here](https://github.com/MinJunKang/DualPixelFace/tree/main/src/module/dcn3d).

[Affine DP Metric](https://github.com/google-research/google-research/tree/master/dual_pixels) : We use this repo to measure performance using affine metric at [here](https://github.com/MinJunKang/DualPixelFace/blob/main/src/metric/affine_dp/metric.py).

Our code is based on [PytorchLightning](https://www.pytorchlightning.ai/).

## References
```
@article{kang2021facial,
  title={Facial Depth and Normal Estimation using Single Dual-Pixel Camera},
  author={Kang, Minjun and Choe, Jaesung and Ha, Hyowon and Jeon, Hae-Gon and Im, Sunghoon and Kweon, In So and Yoon, KuK-Jin},
  journal={arXiv preprint arXiv:2111.12928},
  year={2021}
}
```
