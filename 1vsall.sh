
# kge start models/WNRR18/complex.yaml --job.device cuda:6 --lookup_embedder.dim 40 &
# kge start models/WNRR18/complex.yaml --job.device cuda:5 --lookup_embedder.dim 48 &

# kge start models/WNRR18/complex.yaml --job.device cuda:4 --lookup_embedder.dim 256 &
# kge start models/WNRR18/complex.yaml --job.device cuda:3 --lookup_embedder.dim 512 &
# wait
# kge start models/WNRR18/complex.yaml --job.device cuda:7 --lookup_embedder.dim 1024 &
# kge start models/WNRR18/complex.yaml --job.device cuda:6 --lookup_embedder.dim 2048 &
# kge start models/WNRR18/complex.yaml --job.device cuda:7 --lookup_embedder.dim 32 &
# kge start models/WNRR18/complex.yaml --job.device cuda:5 --lookup_embedder.dim 16 &
# kge start models/WNRR18/complex.yaml --job.device cuda:4 --lookup_embedder.dim 4096 &
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

# # transe
# kge start models/transe.yaml --job.device cuda:0 --lookup_embedder.dim 10 &
# kge start models/transe.yaml --job.device cuda:1 --lookup_embedder.dim 20 &
# kge start models/transe.yaml --job.device cuda:2 --lookup_embedder.dim 40 &
# kge start models/transe.yaml --job.device cuda:3 --lookup_embedder.dim 80 &
# kge start models/transe.yaml --job.device cuda:4 --lookup_embedder.dim 256 &
# wait
# kge start models/transe.yaml --job.device cuda:0 --lookup_embedder.dim 64 &
# kge start models/transe.yaml --job.device cuda:1 --lookup_embedder.dim 128 &
# kge start models/transe.yaml --job.device cuda:2 --lookup_embedder.dim 160 &
# kge start models/transe.yaml --job.device cuda:3 --lookup_embedder.dim 320 &
# kge start models/transe.yaml --job.device cuda:4 --lookup_embedder.dim 640 &
# wait
# # distmult
# kge start models/rotate.yaml --job.device cuda:1 --lookup_embedder.dim 10 &
# kge start models/rotate.yaml --job.device cuda:2 --lookup_embedder.dim 20 &
# kge start models/rotate.yaml --job.device cuda:3 --lookup_embedder.dim 40 &
# kge start models/rotate.yaml --job.device cuda:4 --lookup_embedder.dim 80 &
# kge start models/rotate.yaml --job.device cuda:0 --lookup_embedder.dim 160 &
# wait
# kge start models/rotate.yaml --job.device cuda:0 --lookup_embedder.dim 320 &
# kge start models/rotate.yaml --job.device cuda:1 --lookup_embedder.dim 640 &
# kge start models/rotate.yaml --job.device cuda:2 --lookup_embedder.dim 128 &
# kge start models/rotate.yaml --job.device cuda:3 --lookup_embedder.dim 256 &
# kge start models/rotate.yaml --job.device cuda:4 --lookup_embedder.dim 64 &
# wait
# # conve
# kge start models/conve.yaml --job.device cuda:0 --lookup_embedder.dim 10 &
# kge start models/conve.yaml --job.device cuda:1 --lookup_embedder.dim 20 &
# kge start models/conve.yaml --job.device cuda:2 --lookup_embedder.dim 40 &
# kge start models/conve.yaml --job.device cuda:3 --lookup_embedder.dim 80 &
# kge start models/conve.yaml --job.device cuda:4 --lookup_embedder.dim 160 &
# kge start models/conve.yaml --job.device cuda:0 --lookup_embedder.dim 320 &
# wait
# kge start models/conve.yaml --job.device cuda:1 --lookup_embedder.dim 640 &

# kge start models/conve.yaml --job.device cuda:0 --lookup_embedder.dim 128 &
# kge start models/conve.yaml --job.device cuda:3 --lookup_embedder.dim 256 &
# kge start models/conve.yaml --job.device cuda:4 --lookup_embedder.dim 64 &



## conve, wn18rr
# kge start models/WNRR18/ConvE.yaml --job.device cuda:6 --reciprocal_relations_model.base_model.entity_embedder.dim 512 &
# kge start models/WNRR18/ConvE.yaml --job.device cuda:7 --reciprocal_relations_model.base_model.entity_embedder.dim 256 &
# kge start models/WNRR18/ConvE.yaml --job.device cuda:5 --reciprocal_relations_model.base_model.entity_embedder.dim 128 &

# kge start models/WNRR18/complex.yaml --job.device cuda:4 --lookup_embedder.dim 80 &
# kge start models/WNRR18/complex.yaml --job.device cuda:6 --lookup_embedder.dim 128 &
# kge start models/WNRR18/complex.yaml --job.device cuda:7 --lookup_embedder.dim 256 &
# kge start models/WNRR18/complex.yaml --job.device cuda:5 --lookup_embedder.dim 512 &
# wait
# kge start models/WNRR18/complex.yaml --job.device cuda:4 --lookup_embedder.dim 128  --train.optimizer.default.args.lr 0.2 &
# wait
# kge start models/WNRR18/complex.yaml --job.device cuda:5 --lookup_embedder.dim 128  --train.optimizer.default.args.lr 0.05 &
# kge start models/WNRR18/complex.yaml --job.device cuda:7 --lookup_embedder.dim 128  --train.optimizer.default.args.lr 0.8 &
# wait
# kge start models/WNRR18/complex.yaml --job.device cuda:5 --lookup_embedder.dim 128  --train.loss  bce &
# kge start models/WNRR18/complex.yaml --job.device cuda:6 --lookup_embedder.dim 128  --train.loss  bce_mean &
# kge start models/WNRR18/complex.yaml --job.device cuda:7 --lookup_embedder.dim 128  --train.loss  bce_self_adversarial &
# wait
# kge start models/WNRR18/complex.yaml --job.device cuda:5 --lookup_embedder.dim 128  --train.loss   margin_ranking &
# kge start models/WNRR18/complex.yaml --job.device cuda:6 --lookup_embedder.dim 128  --train.loss   ce &
# kge start models/WNRR18/complex.yaml --job.device cuda:7 --lookup_embedder.dim 128  --train.loss   soft_margin &
# kge start models/WNRR18/complex.yaml --job.device cuda:5 --lookup_embedder.dim 128  --train.loss   se &

## yago-3
# kge start models/YAGO/complex.yaml --job.device cuda:4 --lookup_embedder.dim 256 &
# kge start models/YAGO/complex.yaml --job.device cuda:3 --lookup_embedder.dim 192 &
# kge start models/YAGO/complex.yaml --job.device cuda:2 --lookup_embedder.dim 512 &

# kge start models/YAGO/complex.yaml --job.device cuda:5 &
# kge start models/YAGO/complex.yaml -- job.device cuda:7 &
# wait 
# kge start models/YAGO/complex.yaml -- job.device cuda:4 &
# kge start models/YAGO/complex.yaml -- job.device cuda:5 &
# kge start models/YAGO/complex.yaml -- job.device cuda:6 &
# kge start models/YAGO/complex.yaml -- job.device cuda:7 &


## fb15k-237,complex
# kge start models/fb15k-237/complex.yaml --job.device cuda:5 --train.optimizer.default.args.lr 0.18255  &
# kge start models/fb15k-237/complex.yaml --job.device cuda:6 --train.optimizer.default.args.lr 0.5  &
# kge start models/fb15k-237/complex.yaml --job.device cuda:7 --train.optimizer.default.args.lr 0.65  &


# # ## wn18rr, distmult
# kge start models/WNRR18/distmult.yaml --job.device cuda:1 --lookup_embedder.dim 128 &
# kge start models/WNRR18/distmult.yaml --job.device cuda:2 --lookup_embedder.dim 256 &
# kge start models/WNRR18/distmult.yaml --job.device cuda:3 --lookup_embedder.dim 512 &
# # wait
# kge start models/WNRR18/distmult.yaml --job.device cuda:5 --lookup_embedder.dim 256 --train.type '1vsAll' --train.optimizer.default.args.lr 0.5 &

# wait
# ## fb15k237, distmult
# kge start models/fb15k-237/distmult.yaml --job.device cuda:1 --lookup_embedder.dim 128 &
# kge start models/fb15k-237/distmult.yaml --job.device cuda:2 --lookup_embedder.dim 256 &

# wait
# kge start models/fb15k-237/distmult.yaml --job.device cuda:1 --lookup_embedder.dim 256 --train.type '1vsAll' &
# kge start models/fb15k-237/distmult.yaml --job.device cuda:2 --lookup_embedder.dim 256 --train.type '1vsAll'  --model 'distmult' &
# kge start models/fb15k-237/distmult.yaml --job.device cuda:3 --lookup_embedder.dim 256 --train.type 'KvsAll'  --model 'distmult' &
# distmult，yago3-10

# transe
kge start models/fb15k-237/transe.yaml --job.device cuda:1 --lookup_embedder.dim 128 --train.type '1vsAll' &
kge start models/fb15k-237/transe.yaml --job.device cuda:2 --lookup_embedder.dim 128 --train.type 'KvsAll' &
kge start models/fb15k-237/transe.yaml --job.device cuda:3 --lookup_embedder.dim 128 &
wait
kge start models/WNRR18/transe.yaml --job.device cuda:1 --lookup_embedder.dim 512  &


# rotate
kge start models/fb15k-237/rotate.yaml --job.device cuda:3 --lookup_embedder.dim 256 --train.type '1vsAll' &
wait
kge start models/fb15k-237/rotate.yaml --job.device cuda:3 --lookup_embedder.dim 256 --train.type 'KvsAll' &
kge start models/WNRR18/rotate.yaml --job.device cuda:2 --lookup_embedder.dim 256 --train.type '1vsAll' &
kge start models/WNRR18/rotate.yaml --job.device cuda:1 --lookup_embedder.dim 256  &
# conve
wait
kge start models/YAGO/transe.yaml --job.device cuda:2 --lookup_embedder.dim 128  &
kge start models/YAGO/conve.yaml --job.device cuda:3 --lookup_embedder.dim 256 --train.type '1vsAll' &