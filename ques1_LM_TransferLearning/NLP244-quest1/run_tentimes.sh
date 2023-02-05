#!/bin/sh

a=0
while [ "$a" -lt 3 ]    # this is loop1
do
  echo -n "$a times"
	python main.py
  a=`expr $a + 1`
done
