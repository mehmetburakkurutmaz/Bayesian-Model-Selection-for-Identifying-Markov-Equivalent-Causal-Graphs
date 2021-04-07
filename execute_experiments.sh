#!/bin/bash
START=$1
END=$2
for (( i = $START; i <= $END; i++ ))
do
	if [$i -ne 52] && [$i -ne 53] && [$i -ne 54] && [$i -ne 55] && [$i -ne 71]
	then
    	echo "$i"
	fi
done
