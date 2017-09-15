#!/bin/bash

FILE=opencl_runtime_16.1.1_x64_ubuntu_6.4.0.25

if [[ $TRAVIS_OS_NAME == 'linux' ]]; then
    curl -O http://registrationcenter-download.intel.com/akdlm/irc_nas/9019/${FILE}.tgz
    tar xzf ${FILE}.tgz
    alien ${FILE}/rpm/*.rpm
    sudo dpkg -i *.deb
fi
