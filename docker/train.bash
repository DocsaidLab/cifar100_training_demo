#!/bin/bash

docker run \
    -e USER_ID=$(id -u) \
    -e GROUP_ID=$(id -g) \
    --gpus all \
    --shm-size=64g \
    --ipc=host --net=host \
    --cpuset-cpus="0-31" \
    -v $PWD/cifar100_training_demo:/code/cifar100_training_demo \
    -it --rm cifar100_train bash -c "
echo '
from fire import Fire
from cifar100_training_demo import main_train

if __name__ == \"__main__\":
    Fire(main_train)
' > /code/trainer.py && python /code/trainer.py --cfg_name $1
"