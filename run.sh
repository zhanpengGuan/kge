# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:0 64 0.5 &
# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:0 64 0.05 &
# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:1 256 0.1825 &
python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:1 256 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:2 256 0.2 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:7 256 0.5 &
# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:4 1024 0.1825 &


# python kge/cli_debug.py models/fb15k-237/AdaE_share_rank.yaml cuda:0 '[64,1024]'  0.5 &
# python kge/cli_debug.py models/fb15k-237/AdaE_share_rank.yaml cuda:5 '[64,1024]'  0.1825 &

# python kge/cli_debug.py models/fb15k-237/AdaE_conve.yaml cuda:5 256 0.0042766 &
# python kge/cli_debug.py models/fb15k-237/AdaE_conve.yaml cuda:6 '[256,256]' 0.0042766 &

# # python kge/cli_debug.py models/fb15k-237/AdaE_conve.yaml cuda:7 '[64,1024]' 0.0042766 &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:2 '[1024,1024]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:2 '[128,1024]' 0.5 0.2 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:7 '[256,1024]' 0.15 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:6 '[256,512]' 0.15 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:2 '[256,256]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:4 '[128,1024]' 0.5 0.3 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:4 '[512,256]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:4 '[256,256]' 0.5 0.5 '[0.2]' &

# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:7 '[256,256]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:2 '[64,256]' 0.5 0.0 '[0.5]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:4 '[64,256]' 0.5 0.0 '[0.1]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:7 '[128,1024]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:4 '[64,256,1024]' 0.1825 0.5 '[0.1,0.6]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:5 '[64,256,1024]' 0.5 0.5  '[0.1,0.6]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:3 '[64,256]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:4 '[64,256]' 0.5 0.5 '[0.1]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:5 '[64,256]' 0.5 0.5 '[0.5]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:1 '[64,256]' 0.5 0.5 '[0.95]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:5 '[16,256]' 0.5 0.5 '[0.5]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:5 '[16,256]' 0.5 0.5 '[0.8]' &
# 
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:0 '[128,512]' 0.5 0.3 &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:3 '[128,512]' 0.5 0.2 &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:2 '[128,512]' 0.5 0.1 &



# kge start models/complex.yaml --job.device cuda:0 --lookup_embedder.dim 4096 &
#  kge resume "local/experiments/fb15k-237/20231007-042808-AdaE_rank-rank_zp-[64, 1024]-2048-2048-noBN-0.75-factor-0.75" --job.device cuda:7 &/home/guanzp/code/AdaE/kge/local/experiments/fb15k-237/20231007-091958-AdaE_fix-fix-256-0.5


# distmult
# python kge/cli_debug.py models/fb15k-237/AdaE_distmult.yaml cuda:1 '[1024,256]' 0.2 0.41 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_distmult.yaml cuda:2 '[256,256]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_distmult.yaml cuda:3 '[64,256]' 0.2 0.41 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_distmult.yaml cuda:4 '[64,1024]' 0.2 0.41 '[0.2]' &