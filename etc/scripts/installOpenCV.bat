@echo off
setlocal enabledelayedexpansion

set myRepo=%cd%
set CMAKE_GENERATOR_OPTIONS=-G"Visual Studio 17 2022"
rem set CMAKE_GENERATOR_OPTIONS=-G"Visual Studio 15 2017 Win64"
rem set CMAKE_GENERATOR_OPTIONS=(-G"Visual Studio 16 2019" -A x64)

if not exist "%myRepo%\opencv" (
    echo cloning opencv
    git clone https://github.com/opencv/opencv.git
) else (
    cd opencv
    git pull --rebase
    cd ..
)

if not exist "%myRepo%\opencv_contrib" (
    echo cloning opencv_contrib
    git clone https://github.com/opencv/opencv_contrib.git
) else (
    cd opencv_contrib
    git pull --rebase
    cd ..
)

set RepoSource=opencv
mkdir build_opencv
cd build_opencv

set CMAKE_OPTIONS=-DBUILD_PERF_TESTS:BOOL=OFF -DBUILD_TESTS:BOOL=OFF -DBUILD_DOCS:BOOL=OFF -DWITH_CUDA:BOOL=OFF -DBUILD_EXAMPLES:BOOL=OFF -DINSTALL_CREATE_DISTRIB=ON

echo "************************* %myRepo% -->debug"
cmake %CMAKE_GENERATOR_OPTIONS% %CMAKE_OPTIONS% -DOPENCV_EXTRA_MODULES_PATH="%myRepo%\opencv_contrib\modules" -DCMAKE_INSTALL_PREFIX="%myRepo%\install\%RepoSource%" "%myRepo%\%RepoSource%"
cmake --build . --config debug

echo "************************* %myRepo% -->release"
cmake --build . --config release

cmake --build . --target install --config release
cmake --build . --target install --config debug

cd ..
