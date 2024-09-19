# Define variables
DOCKER_IMAGE_NAME=crm-gpt
DOCKER_CONTAINER_NAME=crm-gpt-container

# Build the Docker image
build:
	docker build -t $(DOCKER_IMAGE_NAME) .

# Run the Docker container
run:
	docker run -d -p 8501:8501 --name $(DOCKER_CONTAINER_NAME) $(DOCKER_IMAGE_NAME)

# Stop and remove the Docker container
stop:
	docker stop $(DOCKER_CONTAINER_NAME)
	docker rm $(DOCKER_CONTAINER_NAME)

# Rebuild the Docker image and run the container
rebuild: stop build run

# Clean up dangling images and containers
clean:
	docker system prune -f
