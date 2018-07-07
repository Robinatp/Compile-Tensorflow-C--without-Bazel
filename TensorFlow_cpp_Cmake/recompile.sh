#!/bin/bash
#
rm -r build/
rm -r bin/
rm -r lib/
mkdir build
cd build
cmake ..
make -j
