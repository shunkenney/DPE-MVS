sudo apt update
sudo apt install cmake
sudo apt install -y libopencv-dev libboost-dev libboost-filesystem-dev libboost-system-dev gdb

mkdir build
cd build
cmake ../DPE-MVS
make
cd ..