#!/bin/bash

pushd ../models

if ! command -v python &> /dev/null
then
    echo "Python is not installed or not in PATH."
    exit 1
fi

python main.py \
  -dataset cifar10 \
  --num-rounds 10000 \
  --eval-every 100 \
  --batch-size 64 \
  --num-epochs 1 \
  --clients-per-round 5 \
  -model resnet9 \
  -lr 0.01 \
  --weight-decay 0.0004 \
  -device cuda:0 \
  -algorithm fedopt \
  --server-lr 1 \
  --server-opt sgd \
  --num-workers 2 \
  --where-loading init \
  -alpha 0.05
