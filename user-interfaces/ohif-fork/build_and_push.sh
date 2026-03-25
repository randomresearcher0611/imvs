#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Define variables
IMAGE_TAG="swasth-frontend-ohif"
DOCKERFILE="Dockerfile_Custom"
TIMESTAMP=$(date +"%Y-%m-%d-%H-%M-%S")
TAR_FILE1="${IMAGE_TAG}---${TIMESTAMP}.tar"
TAR_FILE2="${IMAGE_TAG}---latest.tar"
DESTINATION_ADDRESS_1="?"
DESTINATION_ADDRESS_2="?"

# Build the Docker image
echo "Building Docker image with tag: ${IMAGE_TAG}"
docker build -t ${IMAGE_TAG} -f ${DOCKERFILE} .

# Save the Docker image to the first tar file
echo "Saving Docker image to first tar file: ${TAR_FILE1}"
docker save -o ${TAR_FILE1} ${IMAGE_TAG}

# Save the Docker image to the second tar file
echo "Saving Docker image to second tar file: ${TAR_FILE2}"
docker save -o ${TAR_FILE2} ${IMAGE_TAG}

# Securely copy the first tar file to the first remote server
echo "Copying first tar file to remote server: ${DESTINATION_ADDRESS_1}"
scp ${TAR_FILE1} ${DESTINATION_ADDRESS_1}

# Securely copy the second tar file to the second remote server
echo "Copying second tar file to remote server: ${DESTINATION_ADDRESS_2}"
scp ${TAR_FILE2} ${DESTINATION_ADDRESS_2}

# Print success message
echo "Docker image successfully built, saved, and copied to remote servers."
