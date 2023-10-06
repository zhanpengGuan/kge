
python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:7 '[64,1024]' 0.182 &
python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:1 '[64,1024]' 0.5 &
python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:4 '[64,1024]' 0.75 &
python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:0 '[64,1024]' 1.0 &


# kge start models/complex.yaml --job.device cuda:0 --lookup_embedder.dim 4096 &