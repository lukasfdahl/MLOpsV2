pipeline {
    agent any // Tells Jenkins to run this on any available "worker"

    environment {
        DOCKER_REGISTRY = "kaspersiebrands"  // Docker Hub username
    }

    parameters {
        booleanParam(name: 'RUN_BUILD',         defaultValue: true,  description: 'Build Docker image')
        booleanParam(name: 'RUN_TESTS',         defaultValue: true,  description: 'Run unit tests')
        booleanParam(name: 'RUN_PUSH',          defaultValue: true,  description: 'Push Docker image to Docker Hub')
        booleanParam(name: 'RUN_TRAIN_LOCAL',   defaultValue: true,  description: 'Run local training (sample dataset)')
        booleanParam(name: 'RUN_TRAIN_AILAB',   defaultValue: true,  description: 'Run training on AI-LAB (full dataset)')
        booleanParam(name: 'RUN_POST_TRAINING', defaultValue: true,  description: 'Run post-training optimization on AI-LAB')
        booleanParam(name: 'RUN_DEPLOY',        defaultValue: true,  description: 'Evaluate and deploy model')
        booleanParam(name: 'RUN_DRIFT',         defaultValue: true,  description: 'Run drift detection')
        booleanParam(name: 'RUN_MONITORING',    defaultValue: true,  description: 'Start Prometheus + Grafana monitoring stack')
    }

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
            when { expression { return params.RUN_BUILD } }
            steps {
                echo "Building the Docker container:"
                sh "docker build -f docker/DockerFile -t mlops-kls-container:${env.GIT_COMMIT} ." // Tagged with git commit hash for traceability
            }
        }

        stage("Run Unit Tests") {
            when { expression { return params.RUN_TESTS } }
            steps {
                echo "Running Pytest inside container"
                // To mount the data folder and run the unit tests (${WORKSPACE} is the folder for the current build run)
                sh "docker run --rm -v '${WORKSPACE}/data:/app/data' mlops-kls-container:${env.GIT_COMMIT}"
            }
        }

        stage("Push Docker Image to Registry") {
            when { expression { return params.RUN_PUSH } }
            steps {
                echo "Pushing Docker image to Docker Hub"
                withCredentials([usernamePassword(
                    credentialsId: 'dockerhub-creds',
                    usernameVariable: 'DOCKER_USER',
                    passwordVariable: 'DOCKER_PASS'
                )]) {
                    sh """
                        echo "\$DOCKER_PASS" | docker login -u "\$DOCKER_USER" --password-stdin
                        docker tag mlops-kls-container:${env.GIT_COMMIT} \
                            ${env.DOCKER_REGISTRY}/mlops-kls-container:${env.GIT_COMMIT}
                        docker tag mlops-kls-container:${env.GIT_COMMIT} \
                            ${env.DOCKER_REGISTRY}/mlops-kls-container:latest
                        docker push ${env.DOCKER_REGISTRY}/mlops-kls-container:${env.GIT_COMMIT}
                        docker push ${env.DOCKER_REGISTRY}/mlops-kls-container:latest
                    """
                }
            }
        }

        stage("Model Training Run - Local (sample dataset)") {
            when {
                expression {
                    if (!params.RUN_TRAIN_LOCAL) return false
                    def config = readFile('config/small_train.config.yaml')
                    return config.contains('use_sample_dataset: True')
                }
            }
            steps {
                echo "Starting a local training run with sample dataset"
                sh "docker run --rm --privileged -v ${WORKSPACE}/data:/app/data -e TRAIN_CONFIG=config/small_train.config.yaml mlops-kls-container:${env.GIT_COMMIT} python src/main.py"
            }
        }

        stage("Sync Code to AI-LAB") {
            when {
                expression { return params.RUN_TRAIN_AILAB || params.RUN_POST_TRAINING }
            }
            steps {
                echo "Syncing latest code to AI-LAB"
                sshagent(['ailab-ssh-key']) {
                    sh '''
                        rsync -az -e "ssh -o StrictHostKeyChecking=no" \
                            --filter=':- .gitignore' \
                            --exclude='.git' \
                            --exclude='.dvc/cache' \
                            --exclude='data/dvc' \
                            ./ ksiebr24@student.aau.dk@ailab-fe01.srv.aau.dk:/ceph/project/MLOPS_KLS/
                    '''
                }
            }
        }

        stage("Model Training Run - AI-LAB (full dataset)") {
            when {
                expression {
                    if (!params.RUN_TRAIN_AILAB) return false
                    def config = readFile('config/final_train.config.yaml')
                    return config.contains('ailab_training: True')
                }
            }
            steps { // Fixed: removed extra closing brace
                echo "Submitting training SLURM job"
                sshagent(['ailab-ssh-key']) {
                    sh '''
                        JOB_ID=$(ssh -o StrictHostKeyChecking=no \
                            ksiebr24@student.aau.dk@ailab-fe01.srv.aau.dk \
                            "bash /ceph/project/MLOPS_KLS/slurm/submit_train.sh" \
                            | grep "Submitted batch job" | awk '{print $NF}')
                        echo "Submitted SLURM job: $JOB_ID"

                        for i in $(seq 1 240); do
                            STATUS=$(ssh -o StrictHostKeyChecking=no \
                                ksiebr24@student.aau.dk@ailab-fe01.srv.aau.dk \
                                "squeue -j $JOB_ID -h -o '%T' 2>/dev/null || echo DONE")
                            echo "[$i/240] Job $JOB_ID: $STATUS"
                            [ -z "$STATUS" ] || [ "$STATUS" = "DONE" ] && break
                            sleep 30
                        done
                    '''
                }
            }
        }

        stage("Post-Training Optimization - AI-LAB") {
            when {
                expression {
                    if (!params.RUN_POST_TRAINING) return false
                    def cfg = readFile('config/final_train.config.yaml')
                    return cfg.contains('ailab_training: True')
                }
            }
            steps {
                echo "Submitting post-training SLURM job (quantize + prune + fine-tune)"
                sshagent(['ailab-ssh-key']) {
                    sh '''
                        JOB_ID=$(ssh -o StrictHostKeyChecking=no \
                            ksiebr24@student.aau.dk@ailab-fe01.srv.aau.dk \
                            "sbatch /ceph/project/MLOPS_KLS/slurm/post_training_job.sh" \
                            | awk '{print $NF}')
                        echo "Submitted post-training SLURM job: $JOB_ID"

                        for i in $(seq 1 480); do
                            STATUS=$(ssh -o StrictHostKeyChecking=no \
                                ksiebr24@student.aau.dk@ailab-fe01.srv.aau.dk \
                                "squeue -j $JOB_ID -h -o '%T' 2>/dev/null || echo DONE")
                            echo "[$i/480] Post-train job $JOB_ID: $STATUS"
                            [ -z "$STATUS" ] || [ "$STATUS" = "DONE" ] && break
                            sleep 30
                        done
                    '''
                }
            }
        }

        stage("Drift Detection") {
            when { expression { return params.RUN_DRIFT } }
            steps {
                echo "Running data drift detection on sample dataset"
                sh """
                    docker run --rm \
                        -v ${WORKSPACE}/data:/app/data \
                        -v ${WORKSPACE}/runs:/app/runs \
                        -v ${WORKSPACE}/src:/app/src \
                        --workdir /app \
                        ${env.DOCKER_REGISTRY}/mlops-kls-container:latest \
                        python src/drift.py
                """
                echo "Drift report saved to runs/drift/drift_report.html"
            }
        }

        stage("Start Monitoring Stack") {
            when { expression { return params.RUN_MONITORING } }
            steps {
                echo "Pulling model from DVC and starting monitoring stack"
                sh """
                    mkdir -p ${WORKSPACE}/runs/models
                    # Run dvc pull inside Docker which already has DVC installed
                    docker run --rm \
                        -v ${WORKSPACE}:/app \
                        --workdir /app \
                        ${env.DOCKER_REGISTRY}/mlops-kls-container:latest \
                        dvc pull runs/models/best_model.pth || echo "DVC pull failed, continuing..."

                    docker compose -f docker-compose.monitoring.yml up -d || docker-compose -f docker-compose.monitoring.yml up -d
                    echo "Monitoring stack started:"
                    echo "  Grafana:    http://172.24.198.42:3000  (admin/admin)"
                    echo "  Prometheus: http://172.24.198.42:9090"
                    echo "  API:        http://172.24.198.42:8000/health"
                """
            }
        }

        stage("Evaluate and Deploy Model") {
            when { expression { return params.RUN_DEPLOY } }
            steps {
                echo "Syncing MLflow DB from AI-LAB and evaluating model for deployment"
                sshagent(['ailab-ssh-key']) {
                    // Pull the MLflow DB from AI-LAB so deploy.py can read training results
                    sh '''
                        rsync -az -e "ssh -o StrictHostKeyChecking=no" \
                            ksiebr24@student.aau.dk@ailab-fe01.srv.aau.dk:/ceph/project/MLOPS_KLS/mlflow.db \
                            ${WORKSPACE}/mlflow.db
                    '''
                }
                sh """
                    docker run --rm \
                        -v ${WORKSPACE}/mlflow.db:/app/mlflow.db \
                        -v ${WORKSPACE}/model_card.yaml:/app/model_card.yaml \
                        -v ${WORKSPACE}/src:/app/src \
                        -e MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db \
                        -e MODEL_CARD_PATH=model_card.yaml \
                        --workdir /app \
                        ${env.DOCKER_REGISTRY}/mlops-kls-container:latest \
                        python src/deploy.py
                """
            }
        }
    }

    post {
        always {
            echo "Cleaning up old Docker images:"
            sh "docker rmi mlops-kls-container:${env.GIT_COMMIT} || true" // Cleanup image tagged with commit hash
        }
        success {
            echo "Pipeline passed"
        }
        failure {
            echo "Pipeline failed"
        }
    }
}