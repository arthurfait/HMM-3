#!/bin/bash
cd /home/stuart/Gen_Synthetic_Data/work/

echo "## Parsing output .eval files ##"
mkdir ../OutputResults/

echo "## Merge .pred files ##"
for f in ../io/*.pred;
do
	sed 1d $f > $f.tmp
	cat $f.tmp >> tmp_merged.apv.pred
done

echo '# pos obs hmm_apv match' > merged.apv.pred 
cat tmp_merged.apv.pred >> merged.apv.pred

echo "## EvalTop merged.apv.pred ##"
python evalTop.py merged > ../OutputResults/merged.eval.results


for f in *.eval; 
do
	echo "OutputPaser.py $f ../OutputResults/"
	python OutputParser.py $f ../OutputResults/
done

echo "EvalProtTop.py"
python EvalProtTop.py ../OutputResults/ProtTop.results ../OutputResults/ProtTopEval.results

echo "finished."