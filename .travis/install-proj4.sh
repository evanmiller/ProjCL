#!/bin/bash

PROJ4_FILE=proj-4.9.3

curl -O http://download.osgeo.org/proj/${PROJ4_FILE}.tar.gz
tar xzf ${PROJ4_FILE}.tar.gz
cd ${PROJ4_FILE}
./configure
make
sudo make install
cd ..
