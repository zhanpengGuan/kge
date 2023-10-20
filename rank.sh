# adae_transe
python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:2 '[256,256]' 0.53  '[0.2]' &
python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:0 '[128,128]' 0.53  '[0.2]' & 
python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:1 '[64,64]' 0.53  '[0.2]' & 
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:6 '[24,48]' 0.53  '[0.2]' & 
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:7 '[48,96]' 0.53  '[0.2]' & 
# wait
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:4 '[64,128]' 0.53  '[0.2]' & 
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:5 '[128,256]' 0.53  '[0.2]' & 
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:6 '[256,512]' 0.53  '[0.2]' & 