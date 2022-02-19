cd src
git clone https://github.com/liffiton/PyMiniSolvers.git
cd PyMiniSolvers
make
cd ../..

mkdir data model log


###### Configure pytorch and cuda release verisonn ###############
cd /usr/local/;pwd;ls;rm -rf cuda;ln -s /usr/local/cuda-10.1 /usr/local/cuda;stat cuda
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
nvcc --version
import torch
print(torch.__version__)
