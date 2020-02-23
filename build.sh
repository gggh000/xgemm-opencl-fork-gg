#!/bin/bash
XGEMMHOME=`pwd`

if [ -e ${HOME}/OpenBlas/ ]; then
    echo ${HOME}/OpenBlas/ exists, good
elif [ -e ${XGEMMHOME}/OpenBLAS/ ]; then
    pushd OpenBLAS
      make
      make install PREFIX=${HOME}/OpenBlas
    popd
else
    git clone https://github.com/xianyi/OpenBLAS.git
    pushd OpenBLAS
      make
      make install PREFIX=${HOME}/OpenBlas
    popd
fi
   

 echo "*****************************"
 echo "* BUILDING XGEMM_STRESSTEST *"
 echo "*****************************"



make

