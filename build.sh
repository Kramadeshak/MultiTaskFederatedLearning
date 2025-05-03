#!/bin/sh

echo "***************************************************************"
echo "Building control-plane image"
echo "***************************************************************"

podman build -f ./control-plane/Dockerfile -t control-plane:latest ./control-plane/
echo "control-plane image built successfully"


