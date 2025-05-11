#!/bin/bash

./build.sh

echo "***************************************************************"
echo "Bringing up all the containers"
echo "***************************************************************"
docker-compose up -d
echo "Waiting for the containers to be up"
sleep 5

echo "***************************************************************"
echo "Pushing message to redis"
echo "***************************************************************"
TRAIN_JSON=$(cat test-json/dataset-command.json)
docker exec redis redis-cli PUBLISH commands $TRAIN_JSON
echo "Messaged pushed to redis"

echo "***************************************************************"
echo "Printing control-plane logs"
echo "***************************************************************"
docker logs -f control-plane

echo "***************************************************************"
echo "Printing fedsim-server logs"
echo "***************************************************************"
docker logs -f fedsim-server

echo "***************************************************************"
echo "Printing fedsim-client logs"
echo "***************************************************************"
docker logs -f fedsim-client

echo "***************************************************************"
echo "Bringing down all the containers"
echo "***************************************************************"
docker-compose down -v
