#!/bin/sh

echo "***************************************************************"
echo "Building trainer image"
echo "***************************************************************"

podman build -f ./trainer/Dockerfile -t trainer:latest ./trainer/
echo "trainer image built successfully"


