# Define variables
DOCKER_IMAGE_NAME=crm-gpt
DOCKER_CONTAINER_NAME=crm-gpt-container

# Build the Docker image using Docker Compose
build:
	docker-compose build

# Run the Docker container using Docker Compose
run:
	docker-compose up -d

# Stop and remove the Docker containers using Docker Compose
stop:
	docker-compose down

# Rebuild the Docker image and run the container
rebuild: stop build run

# Clean up dangling images and containers
clean:
	docker system prune -f
