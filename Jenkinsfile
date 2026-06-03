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
        
        stage("Model Training Run - Local (sample dataset)") {
            when {
                expression {
                    def config = readFile('config/small_train.config.yaml')
                    return config.contains('use_sample_dataset: True')
                }
            }
            steps {
                echo "Starting a local training run with sample dataset"
                sh "docker run --rm -v ${WORKSPACE}/data:/app/data mlops-kls-container:${env.BUILD_ID} python src/main.py"
            }
        }

        stage("Model Training Run - AI-LAB (full dataset)") {
            when {
                expression {
                    def config = readFile('config/final_train.config.yaml')
                    return config.contains('use_sample_dataset: False')
                }
            }
            steps {
                echo "Syncing code to AI-LAB and submitting SLURM job"
                sshagent(['ailab-ssh-key']) {
                    sh '''
                        # Push latest code to AI-LAB (skip data/models/cache)
                        rsync -az \
                            --exclude='.git' \
                            --exclude='runs/' \
                            --exclude='.dvc/cache' \
                            --exclude='data/dvc' \
                            ./ ksiebr24@student.aau.dk@ailab-fe01.srv.aau.dk:/ceph/project/MLOPS_KLS/

                        # Submit SLURM job and capture job ID
                        JOB_ID=$(ssh ksiebr24@student.aau.dk@ailab-fe01.srv.aau.dk \
                            "sbatch /ceph/project/MLOPS_KLS/slurm/train_job.sh" \
                            | awk '{print $NF}')
                        echo "Submitted SLURM job: $JOB_ID"

                        # Poll until done (every 30s, max 2h)
                        for i in $(seq 1 240); do
                            STATUS=$(ssh ksiebr24@student.aau.dk@ailab-fe01.srv.aau.dk \
                                "squeue -j $JOB_ID -h -o '%T' 2>/dev/null || echo DONE")
                            echo "[$i/240] Job $JOB_ID: $STATUS"
                            [ -z "$STATUS" ] || [ "$STATUS" = "DONE" ] && break
                            sleep 30
                        done
                    '''
                }
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