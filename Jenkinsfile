pipeline {
/*
 * Defining where to run
 */
//// Any:
// agent any
//// By agent label:
//      agent { label 'sandybridge' }

    agent { label 'Almaha' }
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
                sh "jenkins-scripts/build.sh"
            }
        }
        stage ('test') {
            steps {
                sh '''#!/bin/bash -el
                    # The -x flags indicates to echo all commands, thus knowing exactly what is being executed.
                    # The -e flags indicates to halt on error, so no more processing of this script will be done
                    # if any command exits with value other than 0 (zero)

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
'''
                archiveArtifacts allowEmptyArchive: true, artifacts: 'build/install-dir/starsh.tgz'
            }
        }
        stage ('docs') {
            steps {
                sh "cd $WORKSPACE/build && make docs"
                publishHTML( target: [allowMissing: false, alwaysLinkToLastBuild: false, keepAll: false, reportDir: 'build/docs/html', reportFiles: 'index.html', reportName: 'Doxygen Documentation', reportTitles: ''] )
            }
        }
    }
}
