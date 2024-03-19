
cd cpp
mkdir win_build
cd win_build

set CONAN_REVISIONS_ENABLED=1
conan remote add zscan-conan https://adroco.jfrog.io/artifactory/api/conan/zscan-conan || echo "Ignoring return code of remote add zscan-conan"
conan user %JFROG_CREDS_USR% -r zscan-conan -p %JFROG_CREDS_PSW%
conan install .. --build=missing --profile=../../etc/conan/windows.profile
cmake .. -G "Visual Studio 15 Win64" -DCMAKE_PREFIX_PATH=C:\open3d-0.15.1
cmake --build . --config Release
