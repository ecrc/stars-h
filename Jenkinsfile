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
        }
}
