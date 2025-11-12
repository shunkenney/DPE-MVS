# DPE-MVS
This is python wrapper of DPE-MVS from [https://github.com/ckh0715/DPE-MVS].

# Setup 
## as a project with uv
This guide is aimed for linux users.
### 1. Install DPE-MVS dependencies.
`sudo apt install -y libopencv-dev`
### 2. Install uv
`curl -LsSf https://astral.sh/uv/install.sh | sh`
### 3. Build venv with uv
`uv sync`

## as a python library in uv projects.
### 1. Install DPE-MVS dependencies.
`sudo apt install -y libopencv-dev`
### 2. Add this project as a dependency.
Edit root pyproject.toml as following
```
[project]
dependencies = [
   "DPE_MVS"
]
[tool.uv.sources]
DPE_MVS = { git = "<url/of/this/git-reop>" }
```

## as a C++ build by cmake
```
sudo apt install -y cmake libopencv-dev
mkdir build
cd build
cmake ../csrc/DPE-MVS
make
cd ..
```

# How to call functions.
### Prepare DPE
```
python src/DPE_MVS/colmap2mvsnet.py --dense_folder <path/to/colmap/model/folder> --save_folder <path/to/DPEprepared/folder>
```
### Run DPE as a python library
```
from DPE_MVS import dpe_mvs

dpe_mvs(
   dense_folder="path/to/DPEprepared/folder",  # type: str. 
   gpu_index=0,        # Optional. type: int. default = 0.
   fusion=False,   # Optional. type: bool, default = False. If True, point clouds file (.ply) will be created, which causes overhead.
   viz=False,   # Optional. type: bool, default = False. If True, all visualization info will be saved, which causes overhead.
   depth=True,   # Optional. type: bool, default = False. If True, depth.npy will be created, which causes overhead.
   normal=False,   # Optional. type: bool, default = False. If True, normal.npy will be created, which causes overhead.
   weak=False,   # Optional. type: bool, default = False. If True, weak.npy will be created, which causes overhead.
   edge=False   # Optional. type: bool, default = False. If True, edge.npy will be created, which causes overhead.
)
```
### Run DPE with C++ build
Args and their order are the same as using as python library (above).
```
./build/DPE "path/to/DPEprepared/folder" 0 false false true false false false
```

# Outputs (other than visualization)
### depth.npy  
Depth map.  
shape: [height, width]  
type: np.float32  
### normal.npy
Normal map.  
shape: [height, width, 3]  
type: np.float32  
### weak.npy
Hold the confidence of each pixel. 0 is no confidence (should be discarded), 1 is weak confidence, 2 is strong confidence.  
shape: [height, width]  
type: np.int8  
### edge.npy
Hold edges. 0 is not edge, 1 is on the edge.  
shape: [height, width]  
type: np.int8  
