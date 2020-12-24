#!/bin/bash

curl -O http://registrationcenter-download.intel.com/akdlm/irc_nas/9019/${OPENCL_FILE}.tgz
tar xzf ${OPENCL_FILE}.tgz
sudo alien --scripts -i ${OPENCL_FILE}/rpm/*.rpm
