CMake options {#cmake_opts}
=============

CMake options for STARS-H are gathered automatically, depending on what is
presented in the system. At the end, CMake prints configuration and one can
easily see what is enabled and what is not. Since list of built-in CMake
parameters can be found in its documentation, we present only local STARS-H
options:

    -DOPENMP=OFF

to disable OpenMP in STARS-H. This should be avoided, as many tests rely on
OpenMP function `omp_get_wtime()` as of now.

    -DMPI=OFF

to disable MPI in STARS-H.

    -DSTARPU=OFF

to disable StarPU in STARS-H.

    -DGSL=OFF

to disable Gnu Scientific Library, which is used for spatial statistics
application (matern kernel).

    -DEXAMPLES=OFF

to ignore compilation of examples.

    -DTESTING=OFF

to ignore compilation of tests.

    -DDOCS=OFF

to disable documentation.

    -DDOCS=SHORT

to gather partial documentation only for what will be built and installed.

    -DDOCS=FULL

to gather full documentation.

CMake tests
-----------

In order to run tests, one can use command

    make test

or

    ctest

after finishing compilation to perform check of all possible tests.

Building local documentation
----------------------------

Depending on a value of the option `DOCS`, command

    make docs

will build full/short/no documentation in `build/docs/html` directory in HTML
form.
