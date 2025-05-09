#!/bin/sh

echo "***************************************************************"
echo "Building control-plane image"
echo "***************************************************************"

docker build -f ./control-plane/Dockerfile -t control-plane:latest ./control-plane/
echo "control-plane image built successfully"

echo "***************************************************************"
echo "Building fedsim-server image"
echo "***************************************************************"

docker build -f ./fedsim-server/Dockerfile -t fedsim-server:latest ./fedsim-server/
echo "fedsim-server image built successfully"


echo "***************************************************************"
echo "Building fedsim-client image"
echo "***************************************************************"

docker build -f ./fedsim-client/Dockerfile -t fedsim-client:latest ./fedsim-client/
echo "fedsim-client image built successfully"

