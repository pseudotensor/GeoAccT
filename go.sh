#!/bin/bash

mkdir -p results

#THEANOFLAGS="device=gpu0;floatX=float32"



THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32

python3 main.py

