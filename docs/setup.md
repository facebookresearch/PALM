## Environment Setup

### Simplified Environments for PALM-only

If you only want to use the data and want to use our starter scripts to visualize them, follow the simplified setup instructions below:

```bash
conda create -n palm python=3.10
conda activate palm
cd code
pip install -r requirements_palm.txt
```

Download `MANO_RIGHT.pkl` from [here](https://mano.is.tue.mpg.de/) by registering an account. Then put the pickle in this path:

```bash
code/data/body_models/MANO_RIGHT.pkl
```

### Full Environment Setup

General Requirements (NVIDIA A100):

- PyTorch: 2.0.1+cu118
- Pytorch 3D: 0.7.4
- Kaolin: 0.17.0
- PyTorch Lightning: 1.9.5
- Python: 3.10.14
- CUDA: 11.8 (check `nvcc --version`)

These requirements are non-strict. However, they have been tested on our end. Therefore, it is a good starting point. 

#### Preliminary

**PyTorch Lightning**: To avoid boilerplate code, we use [pytorch lightning (PL)](https://pytorch-lightning.readthedocs.io/en/1.9.5/common/trainer.html) to handle the main logic for training and evaluation. Feel free to consult the documentation, should you have any questions.

Before starting, check your CUDA `nvcc` version:

```bash
nvcc --version # should be 11.8
```

You can install nvcc and cuda via [runfile](https://developer.nvidia.com/cuda-11-8-0-download-archive). If `nvcc --version` is still not `11.8`, check whether you are referring the right nvcc with `which nvcc`. Assuming you have an NVIDIA driver installed, usually, you only need to run the following command to install `nvcc` (as an example):

```bash
sudo bash cuda_*_linux.run --toolkit --silent --override
```

After the installation, make sure the paths pointing to the current cuda toolkit location. For example:

```bash
export CUDA_HOME=/usr/local/cuda-11.8
export PATH="/usr/local/cuda-11.8/bin:$PATH"
export CPATH="/usr/local/cuda-11.8/include:$CPATH"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.8/lib64/"
```

#### PALM-Net environment


```bash
# -- conda
ENV_NAME=palmnet
conda create -n $ENV_NAME python=3.10.14
conda activate $ENV_NAME

cd code

# -- dependencies
pip install -r requirements.txt
pip install pytorch-lightning==1.9.5
conda install -c conda-forge openmesh
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub

# -- torch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

# -- submodules
mkdir submodules && cd submodules

# --- pytroch3d
git clone https://github.com/facebookresearch/pytorch3d.git 
cd pytorch3d
git checkout v0.7.4
export CC=g++-11 # if your GCC has too high version for the next line
export CXX=g++-11 # if your GCC has too high version for the next line
python setup.py install
cd ..

# --- kaolin
pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.1_cu118.html

# --- smplx (custom)
git clone https://github.com/zc-alexfan/smplx.git
cd smplx
git checkout 0d7747f
python setup.py install
cd ../..

# --- tinycuda-nn
pip install --no-build-isolation git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# override numpy
pip install numpy==1.23.5
```

#### Setup files

Download MANO_RIGHT.pkl from [here](https://mano.is.tue.mpg.de/) by registering an account. Then put the pickle in this path: 

```bash
code/data/body_models/MANO_RIGHT.pkl
```

Download our pretrained checkpoint (`flat_hand.right.dim32.7subj.feb.23.pt`) and put it here:

```bash
code/data/pretrained/flat_hand.right.dim32.7subj.feb.23.pt
```

