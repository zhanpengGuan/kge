kge start models/complex.yaml --job.device cuda:7 --lookup_embedder.dim 128 &
# wait
# kge start models/complex.yaml --job.device cuda:1 --lookup_embedder.dim 80 &
# kge start models/complex.yaml --job.device cuda:2 --lookup_embedder.dim 110 &
# kge start models/complex.yaml --job.device cuda:3 --lookup_embedder.dim 210 &
wait
# kge start models/complex.yaml --job.device cuda:1 --lookup_embedder.dim 60 &
# kge start models/complex.yaml --job.device cuda:2 --lookup_embedder.dim 78 &
# kge start models/complex.yaml --job.device cuda:3 --lookup_embedder.dim 30 &
# wait
# kge start models/complex.yaml --job.device cuda:0 --lookup_embedder.dim 10 &
# kge start models/complex.yaml --job.device cuda:1 --lookup_embedder.dim 28 &
# kge start models/complex.yaml --job.device cuda:2 --lookup_embedder.dim 108 &
# kge start models/complex.yaml --job.device cuda:3 --lookup_embedder.dim 40 &
# kge start models/complex.yaml --job.device cuda:4 --lookup_embedder.dim 16 &
# wait
# kge start models/complex.yaml --job.device cuda:5 --lookup_embedder.dim 4096 &