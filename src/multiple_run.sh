#!/bin/sh
i=0
max=20
exec 0<> multi_run_logs.txt
while [ $i -lt $max ]
do
	true $(( i=i+1 ))
	../venv/Scripts/python.exe main.py -m lc -e glove -a train 
	echo $i ": LSTM-CRF with glove done" >&0
	../venv/Scripts/python.exe main.py -m lc -e m2v -a train 
	echo $i ": LSTM-CRF with m2v done" >&0
	../venv/Scripts/python.exe main.py -m lc -e hybrid -a train 
	echo $i ": LSTM-CRF with hybrid done" >&0
	../venv/Scripts/python.exe main.py -m llc -e glove -a train 
	echo $i ": LSTM-LSTM-CRF with glove done" >&0
	../venv/Scripts/python.exe main.py -m llc -e m2v -a train 
	echo $i ": LSTM-LSTM-CRF with m2v done" >&0
	../venv/Scripts/python.exe main.py -m llc -e hybrid -a train 
	echo $i ": LSTM-LSTM-CRF with hybrid done" >&0
    ../venv/Scripts/python.exe main.py -m lcc -e glove -a train 
	echo $i ": CNN-LSTM-CRF with glove done" >&0
	../venv/Scripts/python.exe main.py -m lcc -e m2v -a train 
	echo $i ": CNN-LSTM-CRF with m2v done" >&0
	../venv/Scripts/python.exe main.py -m lcc -e hybrid -a train 
	echo $i ": CNN-LSTM-CRF with hybrid done" >&0
done
exec 0>&-