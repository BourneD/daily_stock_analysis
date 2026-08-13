#!/bin/bash

#set -x

source /root/venv/python_3_13/bin/activate

# rm -f /root/daily_stock_analysis/reports/*

python main.py

cp /root/daily_stock_analysis/reports/* /root/stock_analysis_report/reports/

cd /root/stock_analysis_report
git add .
today=`date +"%Y%m%d"`
git commit -m "report $today"
# git push origin submit_report
