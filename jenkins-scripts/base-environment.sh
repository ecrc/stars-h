# This is meant to be sourced from a sh script

# loads modules
module load gcc/5.5.0
module load mkl/2018-initial
module load openmpi/3.0.0-gcc-5.5.0
module load starpu/1.2.3-gcc-5.5.0-mkl-openmpi-3.0.0
module load gsl/2.4-gcc-5.5.0
module load cmake/3.9.6

# variables
BUILDDIR="$WORKSPACE/build/"
INSTALLDIR="$BUILDDIR/install-dir/"

