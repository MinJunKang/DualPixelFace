#!/bin/bash

mkdir Package
cd Package || exit
git clone https://github.com/NVIDIA/apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
cd ../ || exit

cd src/module/dcn || exit
python setup.py install
cd ../../../ || exit

cd src/module/dcn3d || exit
python setup.py develop
cd ../../../ || exit

pip install -r requirements.txt