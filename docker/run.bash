#!/bin/bash
docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    --gpus all \
    -v $PWD:/code \
    -it --rm cifar100_train