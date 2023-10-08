# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:0 64 0.5 &
# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:0 64 0.05 &
# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:1 256 0.1825 &
# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:3 256 0.5 &
# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:7 256 0.75 &
# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:4 1024 0.1825 &


python kge/cli_debug.py models/fb15k-237/AdaE_share_rank.yaml cuda:0 '[64,1024]'  0.5 &
python kge/cli_debug.py models/fb15k-237/AdaE_share_rank.yaml cuda:5 '[64,1024]'  0.1825 &

# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:1 '[1024,1024]' 0.1825 &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:4 '[1024,1024]' 0.75 &

# kge start models/complex.yaml --job.device cuda:0 --lookup_embedder.dim 4096 &
#  kge resume "local/experiments/fb15k-237/20231007-042808-AdaE_rank-rank_zp-[64, 1024]-2048-2048-noBN-0.75-factor-0.75" --job.device cuda:7 &/home/guanzp/code/AdaE/kge/local/experiments/fb15k-237/20231007-091958-AdaE_fix-fix-256-0.5