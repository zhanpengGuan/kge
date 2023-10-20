# kge start models/complex.yaml --job.device cuda:7 --lookup_embedder.dim 128 &
# wait
# kge start models/complex.yaml --job.device cuda:1 --lookup_embedder.dim 80 &
# kge start models/complex.yaml --job.device cuda:2 --lookup_embedder.dim 110 &
# kge start models/complex.yaml --job.device cuda:3 --lookup_embedder.dim 210 &
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

# transe
kge start models/transe.yaml --job.device cuda:0 --lookup_embedder.dim 10 &
kge start models/transe.yaml --job.device cuda:1 --lookup_embedder.dim 20 &
kge start models/transe.yaml --job.device cuda:2 --lookup_embedder.dim 40 &
kge start models/transe.yaml --job.device cuda:3 --lookup_embedder.dim 80 &
kge start models/transe.yaml --job.device cuda:4 --lookup_embedder.dim 256 &
wait
kge start models/transe.yaml --job.device cuda:0 --lookup_embedder.dim 64 &
kge start models/transe.yaml --job.device cuda:1 --lookup_embedder.dim 128 &
kge start models/transe.yaml --job.device cuda:2 --lookup_embedder.dim 160 &
kge start models/transe.yaml --job.device cuda:3 --lookup_embedder.dim 320 &
kge start models/transe.yaml --job.device cuda:4 --lookup_embedder.dim 640 &
wait
# distmult
kge start models/rotate.yaml --job.device cuda:1 --lookup_embedder.dim 10 &
kge start models/rotate.yaml --job.device cuda:2 --lookup_embedder.dim 20 &
kge start models/rotate.yaml --job.device cuda:3 --lookup_embedder.dim 40 &
kge start models/rotate.yaml --job.device cuda:4 --lookup_embedder.dim 80 &
kge start models/rotate.yaml --job.device cuda:0 --lookup_embedder.dim 160 &
wait
kge start models/rotate.yaml --job.device cuda:0 --lookup_embedder.dim 320 &
kge start models/rotate.yaml --job.device cuda:1 --lookup_embedder.dim 640 &
kge start models/rotate.yaml --job.device cuda:2 --lookup_embedder.dim 128 &
kge start models/rotate.yaml --job.device cuda:3 --lookup_embedder.dim 256 &
kge start models/rotate.yaml --job.device cuda:4 --lookup_embedder.dim 64 &
wait
# conve
kge start models/conve.yaml --job.device cuda:0 --lookup_embedder.dim 10 &
kge start models/conve.yaml --job.device cuda:1 --lookup_embedder.dim 20 &
kge start models/conve.yaml --job.device cuda:2 --lookup_embedder.dim 40 &
kge start models/conve.yaml --job.device cuda:3 --lookup_embedder.dim 80 &
kge start models/conve.yaml --job.device cuda:4 --lookup_embedder.dim 160 &
kge start models/conve.yaml --job.device cuda:0 --lookup_embedder.dim 320 &
wait
kge start models/conve.yaml --job.device cuda:1 --lookup_embedder.dim 640 &
kge start models/conve.yaml --job.device cuda:0 --lookup_embedder.dim 128 &
kge start models/conve.yaml --job.device cuda:3 --lookup_embedder.dim 256 &
kge start models/conve.yaml --job.device cuda:4 --lookup_embedder.dim 64 &
