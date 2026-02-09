@echo off
REM Training script with UTF-8 encoding for Windows compatibility
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
python train.py --folds 3 --strategy re_interp %*
