# kge start models/fb15k-237/complex.yaml --job.device cuda:5  --lookup_embedder.dim 512 --complex.entity_embedder.dropout 0.4 &
# kge start models/fb15k-237/complex.yaml --job.device cuda:6  --lookup_embedder.dim 512 --complex.entity_embedder.dropout 0.3 &
# kge start models/fb15k-237/complex.yaml --job.device cuda:3  --lookup_embedder.dim 512 --complex.entity_embedder.dropout 0.2 &
# wait
# kge start models/fb15k-237/complex.yaml --job.device cuda:3  --lookup_embedder.dim 512 --complex.entity_embedder.dropout 0.6 &
kge start models/fb15k-237/complex.yaml --job.device cuda:5  --lookup_embedder.dim 512 --complex.entity_embedder.dropout 0.8 &
kge start models/YAGO/complex.yaml --job.device cuda:5  --lookup_embedder.dim 512  &
# kge start models/YAGO/complex.yaml --job.device cuda:5  --lookup_embedder.dim 64 &
python kge/cli_debug.py models/YAGO/AdaE_auto.yaml cuda:4 "[1,512]" 0.5 0.1 "[-1]" 512 10 0.01 &