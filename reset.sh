#!/bin/bash
set -e
echo "Reset ac-SpGEMM"
rm -rf build
mkdir build
cd build/
cmake ..
make -j4
echo "Done with setup, ready to run"
echo "For more information please read README.markdown"