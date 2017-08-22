#!/bin/bash -el

# The -x flags indicates to echo all commands, thus knowing exactly what is being executed.
# The -e flags indicates to halt on error, so no more processing of this script will be done
# if any command exits with value other than 0 (zero)

set +x
SCRIPTDIR="$(cd "$(dirname "${0}")"; echo "$(pwd)")"
source $SCRIPTDIR/base-environment.sh
set -x

# Tests
# Set OMP number of threads to number of real cpu cores
# to avoid differences when HT enabled machines.
##nthreads=`lscpu | grep "^Thread" | cut -d: -f 2 | tr -d " "`
#nsockets=`grep "^physical id" /proc/cpuinfo | sort | uniq | wc -l`
#ncorepersocket=`grep "^core id" /proc/cpuinfo | sort  | uniq | wc -l`
#export OMP_NUM_THREADS=$(( nsockets * ncorepersocket ))
# Delete previous CTest results and run tests
rm -rf $BUILDDIR/Testing
cd $BUILDDIR
ctest --no-compress-output -T Test

# archive
cd $INSTALLDIR
rm -f starsh.tgz
tar -zcf starsh.tgz ./*
