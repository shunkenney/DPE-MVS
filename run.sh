python DPE-MVS/colmap2mvsnet.py --dense_folder ../colmap_anno/lane/walking1 --save_folder output_walking1
echo "Finished colmap2mvsnet.py"
./build/DPE output_walking1 0 --no_viz --no_fusion