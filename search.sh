#!/bin/bash

for ratio in "0.3" "0.5" "0.7" "0.9" "-0.3" "-0.5"
do
	for lr in 10 25 50 75
	do
	  for var in "0.03" "0.08" "0.1"
	  do
	   for damp in 0 1 10 100
	   do
	     echo "lr: $lr var: $var damp: $damp ratio: $ratio"
	     python main.py --lr $lr --var $var --damp $damp --ratio $ratio
	     echo "lr: $lr var: $var damp: $damp ratio: $ratio --geo"
	     python main.py --lr $lr --var $var --damp $damp --ratio $ratio --geo
	   done
	  done
	done
done
