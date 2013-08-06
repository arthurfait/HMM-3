#!/bin/bash

echo "## Creating Lr k-fold cv files ##"

for ((i =1; i<=10; i+=1 ));
do
	for ((k =1; k<=10; k+=1 ));
	do	
		if [ $k -ne $i ]; then 
			cat ../sets/set$k >> ../sets/lr$i
		fi
	done
done