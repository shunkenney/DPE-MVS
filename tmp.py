from PIL import Image
import numpy as np

input_file_name = 'output_walking1/DPE/00000000/depth.jpg'
im = np.array(Image.open(input_file_name))
im_gray = np.array(Image.open(input_file_name).convert('L'))
print(im_gray)
print("="*30)
depth = np.load('output_walking1/DPE/00000000/depth.npy')

# Replace nan with 0 for printing statistics
#depth = np.nan_to_num(depth, nan=0.0)
print(depth)
print(np.min(depth), np.max(depth), np.mean(depth), np.median(depth), np.std(depth))
print("="*30)

input_file_name = 'output_walking1/DPE/00000000/weak.jpg'
im = np.array(Image.open(input_file_name))
print(im)
# Print image pixel range for each channel
for i in range(im.shape[2]):
    print(f"Channel {i}: min={np.min(im[:,:,i])}, max={np.max(im[:,:,i])}, mean={np.mean(im[:,:,i])}, median={np.median(im[:,:,i])}, std={np.std(im[:,:,i])}")


print("="*30)
weak = np.load('output_walking1/DPE/00000000/weak.npy')

# Replace nan with 0 for printing statistics
#depth = np.nan_to_num(depth, nan=0.0)
print(weak)
print(np.min(weak), np.max(weak), np.mean(weak), np.median(weak), np.std(weak))
print(weak.dtype)
print("="*30)