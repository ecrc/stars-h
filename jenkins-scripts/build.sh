#!/bin/bash -xel

# The -x flags indicates to echo all commands, thus knowing exactly what is being executed.
# The -e flags indicates to halt on error, so no more processing of this script will be done
# if any command exits with value other than 0 (zero)

set +x
SCRIPTDIR="$(cd "$(dirname "${0}")"; echo "$(pwd)")"
source $SCRIPTDIR/base-environment.sh
set -x
# initialise git submodule
git submodule update --init
mkdir -p $BUILDDIR && cd $BUILDDIR && rm -rf ./CMake*
cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALLDIR # -DMPIEXEC=$(which mpirun)

# Build only 
make

# Install
make install
