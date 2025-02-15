#这里是p的超参数实验
python kge/cli_debug.py models/WNRR18/AdaE_auto.yaml cuda:7 "[1,256]" 0.5 0.1 "[-1]" 256 4 0.01 &
python kge/cli_debug.py models/WNRR18/AdaE_auto.yaml cuda:7 "[1,256]" 0.5 0.1 "[-1]" 256 5 0.01 &