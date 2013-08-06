#!/bin/bash

cd /home/redpath/Synthetic_Data/work/

ws=7
for ((i =1; i<=10; i+=1 ));
do
    echo "## train$i ## ws$ws"
    echo "preprocess.py $ws ../sets/lr$i ../sets/set$i ../seq_profiles/ ../io/lr$i/pca_seq_profiles/"
    mkdir ../io/lr$i/
	mkdir ../io/lr$i/pca_seq_profiles/
	python preprocess.py $ws ../sets/lr$i ../sets/set$i ../seq_profiles/ ../io/lr$i/pca_seq_profiles/
	echo "profile_clustering.py ../sets/lr$i ../io/lr$i/pca_seq_profiles/ ../io/lr$i/"
	python profile_clustering.py ../sets/lr$i ../io/lr$i/pca_seq_profiles/ ../io/lr$i/
    echo "protein_GMMHMM.py ../sets/lr$i ../io/lr$i/pca_seq_profiles/ ../io/lr$i"
	python protein_GMMHMM.py ../sets/lr$i ../io/lr$i/pca_seq_profiles/ ../io/lr$i > results$i.out
    
    echo "## Test set$i ## lr$i.mod.final"
    for f in `cat ../sets/set$i` 
    do
    	echo "pred predvit.py ../io/lr$i/pca_seq_profiles/$f ../io/lr$i"
    	echo ">" $f
    	python predvit.py ../io/lr$i/pca_seq_profiles/$f ../io/lr$i
    	python evalTop.py ../io/$f > set$i.$f.eval
    done
done
echo "finished."
