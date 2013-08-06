#!/bin/bash
cd /home/stuart/Gen_Synthetic_Data/work/
i=1
#do
    echo "## train$i ##"
    echo "profile_clustering.py ../sets/lr$i ../gen_seq_profiles/ ../io/lr$i/"
	python profile_clustering.py ../sets/lr$i ../gen_seq_profiles/ ../io/lr$i/ > profile_clustering$i.out
    echo "protein_GMMHMM.py ../sets/lr$i ../gen_seq_profiles/ ../io/lr$i"
	python protein_GMMHMM.py ../sets/lr$i ../gen_seq_profiles/ ../io/lr$i > results$i.out
    
    echo "## Test set$i ## lr$i.mod.final"
    for f in `cat ../sets/set$i` 
    do
    	echo "pred predvit.py ../gen_seq_profiles/$f ../io/lr$i"
    	echo ">" $f
    	python predvit.py ../gen_seq_profiles/$f ../io/lr$i
    	python evalTop.py ../io/$f > set$i.$f.eval
    done
#done
echo "finished."
