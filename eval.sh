#!/bin/bash
ROLLNO=$1
jupyter nbconvert --to script "${ROLLNO}_foml24_hackathon.ipynb"
python "${ROLLNO}_foml24_hackathon.py" --train-file train.csv --test-file test.csv --predictions-file ${ROLLNO}_submission.csv
