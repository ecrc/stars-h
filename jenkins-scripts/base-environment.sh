# This is meant to be sourced from a sh script

# loads modules
module load gcc/5.3.0 openblas/0.2.19-gcc-5.3.0-openmp mpi-openmpi/2.1.0-gcc-5.3.0
module load cmake/3.3.2

# variables
BUILDDIR="$WORKSPACE/build/"
INSTALLDIR="$BUILDDIR/install-dir/"

