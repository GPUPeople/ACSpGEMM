#!/bin/bash
set -e
echo "Setup ac-SpGEMM"
if [ ! -d "include/external" ]; then
  echo "Clone CUB into include/externals"
  cd include && mkdir external && cd external && git clone https://github.com/NVlabs/cub.git . && cd ../..
fi
echo "Setup build folder, run CMake and build"
if [ ! -d "build" ]; then
  mkdir build
fi
chmod +x test/runall.sh
cd build/
export CUDACXX="/usr/local/cuda/bin/nvcc"
cmake ..
make -j4
echo "Done with setup, ready to run"
echo "For more information please read README.markdown"