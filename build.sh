#!/bin/sh

echo "***************************************************************"
echo "Building control-plane image"
echo "***************************************************************"

podman build -f ./control-plane/Dockerfile -t control-plane:latest ./control-plane/
echo "control-plane image built successfully"

echo "***************************************************************"
echo "Building fedsim-server image"
echo "***************************************************************"

podman build -f ./fedsim-server/Dockerfile -t fedsim-server:latest ./fedsim-server/
echo "fedsim-server image built successfully"


