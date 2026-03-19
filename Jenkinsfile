pipeline {
    agent any // Tells Jenkins to run this on any available "worker"

    stages {
        stage("Checkout") {
            steps {
                // Jenkins automatically clones the GitHub repo here
                checkout scm
            }
        }

        stage("Build Docker Image") {
            steps {
                echo "Building the Docker container:"
                sh "docker build -f docker/Dockerfile -t mlops-kls-container:${env.BUILD_ID} ."
            }
        }

        stage("Run Unit Tests") {
            steps {
                echo "Running Pytest inside container"
                // To mount the data folder and run the unit tests
                sh "docker run --rm -v '$(pwd)/data:/app/data' mlops-kls-container:${env.BUILD_ID}"
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
