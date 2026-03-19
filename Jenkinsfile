pipeline {
    agent any // Tells Jenkins to run this on any available "worker"

    options {
        timestamps() // Adds clock times to the logs
    }

    stages {
        stage("Checkout") {
            steps {
                // Jenkins automatically clones the GitHub repo here
                checkout([$class: 'GitSCM', 
                branches: [[name: '*/development']], 
                userRemoteConfigs: [[url: 'https://github.com/lukasfdahl/MLOpsV2.git', 
                credentialsId: 'github-kls-bot']]])
            }
        }

        stage("Build Docker Image") {
            steps {
                echo "Building the Docker container:"
                sh "docker build -f docker/DockerFile -t mlops-kls-container:${env.BUILD_ID} ."
            }
        }

        stage("Run Unit Tests") {
            steps {
                echo "Running Pytest inside container"
                // To mount the data folder and run the unit tests (${WORKSPACE} is the folder for the current build run)
                sh "docker run --rm -v '${WORKSPACE}/data:/app/data' mlops-kls-container:${env.BUILD_ID}"
            }
        }
        stage("Run Model Training Run") {
            steps {
                echo "Starting a training run"
                sh "docker run --rm -v ${WORKSPACE}/data:/app/data mlops-kls-container:${env.BUILD_ID} python src/main.py"
            }
        }
    }

    post {
        always {
            echo "Cleaning up old Docker images:"
            sh "docker rmi mlops-kls-container:${env.BUILD_ID} || true"
        }
        success {
            echo "Pipeline passed"
        }
        failure {
            echo "Pipeline failed"
        }
    }
}
