#!/bin/bash

OPENCL_FILE=opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25

if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
    curl -O http://registrationcenter-download.intel.com/akdlm/irc_nas/9019/${OPENCL_FILE}.tgz
    tar xzf ${OPENCL_FILE}.tgz
    sudo alien --scripts -i ${OPENCL_FILE}/rpm/*.rpm
fi
