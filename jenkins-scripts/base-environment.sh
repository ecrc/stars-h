# This is meant to be sourced from a sh script

# loads modules
module load gcc/5.3.0
module load mkl/2018-initial
module load mpi-openmpi/1.10.2-gcc-5.3.0
module load starpu/1.2.1-gcc-5.3.0
module load gsl/2.3-gcc-5.3.0
module load cmake/3.3.2

# variables
BUILDDIR="$WORKSPACE/build/"
INSTALLDIR="$BUILDDIR/install-dir/"

