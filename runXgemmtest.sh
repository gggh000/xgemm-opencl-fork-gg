export LD_LIBRARY_PATH=${HOME}/OpenBlas/lib:$LD_LIBRARY_PATH;
echo $LD_LIBRARY_PATH
./xgemmStandaloneTest;
