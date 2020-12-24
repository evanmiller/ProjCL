#!/bin/bash

curl -O http://download.osgeo.org/proj/${PROJ4_FILE}.tar.gz
tar xzf ${PROJ4_FILE}.tar.gz
cd ${PROJ4_FILE}
cmake . -DBUILD_GEOD=OFF -DBUILD_CCT=OFF -DBUILD_CS2CS=OFF -DBUILD_TESTING=OFF -DBUILD_GIE=OFF -DBUILD_PROJ=OFF -DBUILD_PROJINFO=OFF -DBUILD_PROJSYNC=OFF -DENABLE_CURL=OFF
make
sudo make install
cd ..
