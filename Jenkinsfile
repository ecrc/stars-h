pipeline {
/*
 * Defining where to run
 */
//// Any:
// agent any
//// By agent label:
//      agent { label 'sandybridge' }

    agent { label 'jenkinsfile' }
    triggers {
        pollSCM('H/10 * * * *')
    }
    environment {
        XX="gcc"
    }

    options {
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '50'))
        timestamps()
    }

    stages {
        stage ('build') {
            steps {
                sh '''#!/bin/bash -el
                    # The -x flags indicates to echo all commands, thus knowing exactly what is being executed.
                    # The -e flags indicates to halt on error, so no more processing of this script will be done
                    # if any command exits with value other than 0 (zero)

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

                    set -x
                    # initialise git submodule
                    git submodule update --init
                    mkdir -p $BUILDDIR && cd $BUILDDIR && rm -rf ./CMake*
                    cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALLDIR # -DMPIEXEC=$(which mpirun)

                    # Build only
                    make

                    # Install
                    make install
'''
            }
        }
        stage ('test') {
            steps {
                sh '''#!/bin/bash -el
                    # The -x flags indicates to echo all commands, thus knowing exactly what is being executed.
                    # The -e flags indicates to halt on error, so no more processing of this script will be done
                    # if any command exits with value other than 0 (zero)

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
                    if [ "master" == "$BRANCH_NAME" ]
                    then
                        # if master branch, run full set of tests
                        ctest --no-compress-output -T Test
                    else
                        # Run every fourth test
                        ctest --no-compress-output -T Test -I 1,,4
                    fi

                    # archive
                    cd $INSTALLDIR
                    rm -f starsh.tgz
                    tar -zcf starsh.tgz ./*
'''
                archiveArtifacts allowEmptyArchive: true, artifacts: 'build/install-dir/starsh.tgz'
            }
        }
        stage ('docs') {
            steps {
                sh "cd $WORKSPACE/build && make docs"
                sh '''#!/bin/bash -ex
                      cd $WORKSPACE
                      rm -rf cppcheckhtml
                      cppcheck --enable=all --xml --xml-version=2 testing/ src/ examples/ -I include/   2> cppcheck.xml
                      cppcheck-htmlreport --source-encoding="iso8859-1" --title="STARS-H" --source-dir=. --report-dir=cppcheckhtml --file=cppcheck.xml
'''
                publishHTML( target: [allowMissing: false, alwaysLinkToLastBuild: false, keepAll: false, reportDir: 'build/docs/html', reportFiles: 'index.html', reportName: 'Doxygen Documentation', reportTitles: ''] )
                publishHTML( target: [allowMissing: false, alwaysLinkToLastBuild: false, keepAll: false, reportDir: 'cppcheckhtml', reportFiles: 'index.html', reportName: 'CppCheckReport', reportTitles: ''] )
            }
        }
    }
}
