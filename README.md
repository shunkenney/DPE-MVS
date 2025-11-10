# DPE-MVS

## Setup with `uv`
1. Make sure system dependencies such as CUDA, Boost (filesystem/system), and OpenCV are installed (see `setup.sh` for apt packages that are still required).
2. From the project root run:

   ```bash
   uv sync
   ```

   This creates a `.venv`, installs the build toolchain (`cmake`, `ninja`, `pybind11`, `scikit-build-core`), and builds the `_dpe` extension in editable mode.

3. Use the environment for any tooling:

   ```bash
   # Convert COLMAP output if needed
   uv run python python/DPE_MVS/colmap2mvsnet.py \
     --dense_folder ../colmap_anno/lane/walking1 \
     --save_folder output

   # Run the pipeline directly from Python
   uv run python -c "from DPE_MVS import dpe_mvs; dpe_mvs('output', gpu_index=0, fusion=False, viz=False, weak=False)"
   ```

If you prefer the legacy binary interface, build with CMake inside the synced environment and run `./build/DPE` exactly as before:

```
./build/DPE output_walking1 0 --no_viz --no_fusion --no_weak
```

Args:
  1st. input folder path  
  2nd. GPU index  
  - `--no_viz`: If provided, some visualization jpg files will not be created.  
  - `--no_fusion`: If provided, point-cloud `.ply` files will not be created.  
  - `--no_weak`: If provided, `weak.jpg` will not be created.
