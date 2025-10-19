<p align="center">
<h1 align="center">Easy3D: 
A Simple Yet Effective Method for 3D Interactive Segmentation</h1>
<p align="center">
<a href="https://simonelli-andrea.github.io/"><strong>Andrea Simonelli</strong></a>, <a href="https://sirwyver.github.io/"><strong>Norman Müller</strong></a>, <a href="https://scholar.google.it/citations?user=CxbDDRMAAAAJ&hl=en"><strong>Peter Kontschieder</strong></a>
</p>
<h3 align="center">ICCV 2025 (Oral)</h3>
<h3 align="center"><a href="https://arxiv.org/pdf/2504.11024"><strong>Paper</strong></a> | <a href="https://simonelli-andrea.github.io/easy3d/"><strong>Project</strong></a></h3>
</p>

<br><br>

## Setup

The repo requires a minimal setup, all the requirements can be easily installed with pip.

### Create a conda environment
```
conda create --name easy3d python=3.11
conda activate easy3d
```
Note that we need to use python<=3.11 due to spconv. If a new spconv version supports python >=3.12 use that.

### Install pytorch
```
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
```
Select the CUDA version according to your hardware. The code probably works with newer versions of pytorch as well.

### Install spconv
```
pip install spconv-cu121
```
Note that the spconv version does not necessarily have to match your CUDA version. We used cu121 with CUDA 12.8 without issues.


### Install additional packages
```
pip install pyyaml tensorboard plyfile
```

### Install easy3d
```
pip install -e .
```

### [Optional] Install packages for the demo
```
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install trimesh viser nerfview matplotlib splines jaxtyping
```

<br><br>

## Training

The repository has been explicitly kept simple and minimalistic. Launching the training should be easy and fast.

### Download data
Download and unzip the preprocessed data from the gdrive folder [here](https://drive.google.com/file/d/1cqWgVlwYHRPeWJB-YJdz-mS5njbH4SnG/view) (from AGILE3D), or follow:
```
pip install gdown
gdown 1cqWgVlwYHRPeWJB-YJdz-mS5njbH4SnG
unzip data.zip
```
Move the data to /data and or change the "data_root" in the config accordingly to the data location.

Note that this data has been created by the authors of [AGILE3D](https://ywyue.github.io/AGILE3D/), and contains a preprocessed ScanNet40, S3DIS and KITTI-360. All the data assumes to have the up direction as +Z and gravity (down-vector) as -Z, as in ScanNet40.

### Train
Define the number of GPUs and the experiment output path, then launch the training:
```
export NUM_GPUS=4
export EXPERIMENT_DIR="experiments/debug"
export OMP_NUM_THREADS=8  # adjust if needed
torchrun --nnodes=1 --nproc_per_node=$NUM_GPUS scripts/train.py configs/default.yaml --work_dir=$EXPERIMENT_DIR
```
We trained our model with at least 4GPUs.

### Log
You can check the training with tensorboard:
```
export TB_PORT=8002
tensorboard --logdir=$EXPERIMENT_DIR --port=$TB_PORT
```

<br><br>

## Demo

We provide an interactive Viser demo, which allows to run a pretrained segmentation model on a mesh (PLY). With proper port forwarding, Viser allows to run the demo even on a remote host. 

You can download a pretrained model from [here](https://github.com/facebookresearch/easy3d/releases/download/v1.0/pretrained_easy3d.pth).

```
python scripts/demo.py --mesh=/path/to/mesh.ply --config=configs/default.yaml --ckpt=/path/to/pretrained_easy3d.pth
```
Use the buttons on the right to define positive or negative clicks, then run the segmentation and see the results directly on the mesh.

**Note:** mesh must be oriented with the up direction = +Z, as in ScanNet40.

<br><br>


## License

The model is licensed under the [CC BY-NC](LICENSE).

## Contributing

See [contributing](CONTRIBUTING.md) and the [code of conduct](CODE_OF_CONDUCT.md).


<br><br>


## Citing Easy3D

If you use Easy3D in your research, please use the following entry:

```
@article{simonelli2025easy3d,
  author    = {Simonelli, Andrea and M{\"u}ller, Norman and Kontschieder, Peter},
  title     = {Easy3D: A Simple Yet Effective Method for 3D Interactive Segmentation},
  journal   = {ICCV},
  year      = {2025},
}
```

<br><br>
