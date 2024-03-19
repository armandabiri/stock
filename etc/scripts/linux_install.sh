#!/bin/bash
# exit when any command fails
set -e

BASEDIR=$(dirname $0)
cd $BASEDIR

INSTALLDIR=/opt/adroco/adroco/

# Copy program to the adroco folder
mkdir -p $INSTALLDIR
cp -R . $INSTALLDIR
chmod 777 $INSTALLDIR
