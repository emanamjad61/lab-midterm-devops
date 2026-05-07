pipeline {
    agent any

    options {
        disableConcurrentBuilds()
        timestamps()
    }

    stages {
        stage('Fetch Data From GitHub') {
            steps {
                // This step clones the repository specified in the Jenkins Job configuration
                checkout scm
            }
        }

        stage('Train Model') {
            steps {
                // Runs training inside a temporary python container to ensure clean environment
                sh '''
                    docker run --rm \
                      -v "$PWD":/app \
                      -w /app \
                      python:3.11-slim \
                      sh -c "pip install --no-cache-dir -r requirements.txt && python train.py"
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                // Builds the final production image for the FastAPI app
                sh 'docker build -t ml-midterm-api:latest .'
            }
        }

        stage('Run Docker Container') {
            steps {
                // Removes old container if exists and runs the new one
                sh '''
                    docker rm -f ml-midterm-api || true
                    docker run -d \
                      --name ml-midterm-api \
                      --restart unless-stopped \
                      -p 8000:8000 \
                      ml-midterm-api:latest
                '''
            }
        }

        stage('Verify Metrics API') {
            steps {
                // Waits for startup and verifies the /metrics endpoint is reachable
                sh '''
                    sleep 5
                    curl -f http://localhost:8000/metrics
                '''
            }
        }
    }
}
