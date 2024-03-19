#!/bin/bash
# exit when any command fails
set -e

echo "Releasing from $BRANCH_NAME"
echo "Release $TAG_NAME"

# Todo(Andrei): Move this to docker
sudo apt-get install zip -y

cd cpp/build
cmake --install . --prefix adroco
zip -r adroco.zip adroco/

curl -u $JFROG_CREDS_USR:$JFROG_CREDS_PSW -T adroco.zip "https://adroco.jfrog.io/artifactory/zscan-main-generic-local/adroco/$TAG_NAME.zip"
