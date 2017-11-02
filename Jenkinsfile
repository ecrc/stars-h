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
                sh "jenkins-scripts/test.sh"
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
