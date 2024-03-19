#!/bin/bash
# exit when any command fails
set -e

git config --global --add safe.directory $PWD

cd cpp
mkdir -p build || echo 0
cd build
export CONAN_REVISIONS_ENABLED=1
conan install .. --lockfile=../linux-conan.lock --update
cmake .. -D_GLIBCXX_USE_CXX11_ABI=1 -DCMAKE_PREFIX_PATH=/opt/open3d/
cmake --build . --config Release
