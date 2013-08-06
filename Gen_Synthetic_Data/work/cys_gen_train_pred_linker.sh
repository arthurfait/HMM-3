#!/bin/bash
cd /home/stuart/Gen_Synthetic_Data/work/

for ((i =1; i<=10; i+=1 ));
do
	echo "## cys_gmmhmm_gen_$i.sh ##"
    ./cys_gmmhmm_gen_$i.sh
done

echo "## create_lr.sh ##"
./create_lr.sh

for ((i =1; i<=10; i+=1 ));
do
	echo "## cys_gmmhmm_train_pred_$i.sh ##"
    ./cys_gmmhmm_train_pred_$i.sh
done

echo "## OutputParser.sh ##"
./OutputParser.sh

echo "finished."
