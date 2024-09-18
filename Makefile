# Define variables
DOCKER_IMAGE_NAME=my_multi_agent_app
DOCKER_CONTAINER_NAME=my_multi_agent_app_container

# Build the Docker image
build:
	docker build -t $(DOCKER_IMAGE_NAME) .

# Run the Docker container
run:
	docker run -d -p 8000:8000 -p 8501:8501 --name $(DOCKER_CONTAINER_NAME) $(DOCKER_IMAGE_NAME)

# Stop and remove the Docker container
stop:
	docker stop $(DOCKER_CONTAINER_NAME)
	docker rm $(DOCKER_CONTAINER_NAME)

# Rebuild the Docker image and run the container
rebuild: stop build run
