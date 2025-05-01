#!/bin/bash

./build.sh

echo "***************************************************************"
echo "Bringing up all the containers"
echo "***************************************************************"
podman-compose up -d
echo "Waiting for the containers to be up"
sleep 5

echo "***************************************************************"
echo "Pushing message to redis"
echo "***************************************************************"
TRAIN_JSON=$(cat test-json/trainer_data.json)
podman exec redis redis-cli PUBLISH trainer_commands $TRAIN_JSON
echo "Messaged pushed to redis"

echo "***************************************************************"
echo "Printing trainer logs"
echo "***************************************************************"
podman logs -f trainer

echo "***************************************************************"
echo "Bringing down all the containers"
echo "***************************************************************"
podman-compose down -v
