#!/bin/bash
cd /home/stuart/Gen_Synthetic_Data/work/
i=10
dim=4
max_len=10
num_seqs=2000

echo "## Gen_Seq$i ## Dim$dim"
echo "gen_GMMHMM.py lr$i ../io/lr$i/ $dim $max_len $num_seqs ../gen_seq_profiles/ set$i"
mkdir ../io/lr$i/
mkdir ../gen_seq_profiles/
python gen_GMMHMM.py lr$i ../io/lr$i/ $dim $max_len $num_seqs ../gen_seq_profiles/ set$i > gen_results$i.out
    
echo "generation finished."
