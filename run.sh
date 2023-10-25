# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:0 64 0.5 &
# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:0 64 0.05 &
# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:1 256 0.1825 &
# python kge/cli_debug.py models/fb15k-237/AdaE_fix.yaml cuda:1 256 0.5 0.5 '[0.2]' &
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
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:5 '[256,1024]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:2 '[128,1024]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:4 '[128,1024]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:4 '[512,256]' 0.5 0.5 '[0.2]' &



# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:0 '[128,256,1024]' 0.5 0.5 '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:2 '[64,256,512]' 0.5 0.5 '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:4 '[128,256,512]' 0.5 0.5 '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]' &
# wait
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:1 '[64,256,1024]' 0.5 0.5 '[0.02,0.04,0.08,0.16,0.32,0.64]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:3 '[128,256,1024]' 0.5 0.5 '[0.02,0.04,0.08,0.16,0.32,0.64]' &
# wait
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:2 '[1024,256,64]' 0.5 0.5 '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:6 '[64,1024,256]' 0.5 0.5 '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]' &



# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:7 '[64,256,1024]' 0.5 0.5 '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:0 '[128,512]' 0.5 0.5 '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:2 '[64,1024]' 0.5 0.5 '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]' &
# wait
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:2 '[16,64,512,1024]' 0.5 0.5 '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:4 '[64,128,512,1024]' 0.5 0.5 '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:4 '[64,512]' 0.5 0.5 '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:7 '[128,256,1024]' 0.5 0.5 '[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:2 '[64,256,1024,2048]' 0.5 0.5 '[0.04,0.2,0.5]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:0 '[256,512]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:2 '[128,256,512]' 0.5 0.5 '[0.1,0.6]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:3 '[64,256,512]' 0.5 0.5 '[0.1,0.5]' &





# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:0 '[16,32]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:2 '[48,96]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:3 '[48,192]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:4 '[24,48]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:5 '[24,96]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:6 '[8,16]' 0.5 0.5 '[0.2]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:4 '[512,1024,2048]' 0.5 0.5 '[0.2,0.5]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:5 '[256,1024,2048]' 0.5 0.5 '[0.2,0.5]' &

# python kge/cli_debug.py models/fb15k-237/AdaE_rank.yaml cuda:4 '[512,1024]' 0.5 0.5 '[0.2]' &
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

# python kge/cli_debug.py models/fb15k-237/transe_rank.yaml cuda:5 '[8,16]' 0.0003 0.5 '[0.2]' &

# # 细粒度
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:1 '[64,256,1024]' 0.5 0.5 '[-1]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:2 '[64,256,512]' 0.5 0.5 '[-1]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:3 '[128,256,512]' 0.5 0.5 '[-1]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:4 '[8,16]' 0.5 0.5 '[-1]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:5 '[8,16,32]' 0.5 0.5 '[-1]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:7 '[4,8,16,32]' 0.5 0.5 '[-1]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:7 '[10,30,50]'  0.5 0.5 '[-1]' 50 3 0.01 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:5 '[10,30,50]'  0.5 0.5 '[-1]' 30 3 0.01 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:1 '[64,128,256]'  0.5 0.5 '[-1]' 256 3 0.01 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:2 '[64,128,256]'  0.5 0.5 '[-1]' 256 5 0.01 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:3 '[64,128,256]'  0.5 0.5 '[-1]' 256 1 0.01 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:4 '[64,128,256]'  0.5 0.5 '[-1]' 256 3 0.001 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:5 '[64,128,256]'  0.5 0.5 '[-1]' 256 5 0.001 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:6 '[64,128,256]'  0.5 0.5 '[-1]' 256 1 0.001 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:7 '[64,128,256]'  0.5 0.5 '[-1]' 256 3 0.1 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:0 '[64,128,256]'  0.5 0.5 '[-1]' 256 5 0.1 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:0 '[64,128,256]'  0.5 0.5 '[-1]' 256 1 0.1 &


# ## YAGO-3
# python kge/cli_debug.py models/YAGO/AdaE_rank.yaml cuda:4 "[64,128]" 0.5 0.0 "[0.2]" 128 &
# python kge/cli_debug.py models/YAGO/AdaE_rank.yaml cuda:6 "[128,128]" 0.5 0.0 "[0.2]" 128 &

### WN18RR
## 学习率
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:1 "[64,128]" 0.25 0.34 "[0.2]" 128 &
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:2 "[64,128]" 0.5 0.34 "[0.2]" 128 &
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:3 "[64,128]" 0.75 0.34 "[0.2]" 128 &
# wait
# ## 同参数
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:1 "[64,384]" 0.5 0.34 "[0.2]" 128 &
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:2 "[64,384]" 0.75 0.34 "[0.2]" 128 &
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:3 "[64,384]" 0.5 0.34 "[0.2]" 384 &
## transform layer学习率实验
# python kge/cli_debug.py models/WN18RR/AdaE_rank.yaml cuda:2 "[64,128]" 0.5 0.34 "[0.2]" 128 &

## complEx而不是AdaE
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:5 "[64,384]" 0.5 0.36 "[0.2]" 128 &
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:1 "[40,80]" 0.526 0.36 "[0.5]" 80 &
python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:1 "[40,80]" 0.526 0.36 "[0.1]" 80 &
python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:2 "[40,80]" 0.526 0.36 "[0.3]" 80 &
python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:3 "[40,80]" 0.526 0.36 "[0.4]" 80 &
python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:4 "[40,80]" 0.526 0.36 "[0.6]" 80 &
python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:1 "[40,80]" 0.526 0.36 "[0.7]" 80 &
python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:2 "[40,80]" 0.526 0.36 "[0.8]" 80 &
python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:3 "[40,80]" 0.526 0.36 "[0.9]" 80 &
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:0 "[64,80]" 0.526 0.36 "[0.2]" 80 &
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:3 "[48,80]" 0.526 0.36 "[0.2]" 80 &
# python kge/cli_debug.py models/WNRR18/AdaE_rank.yaml cuda:2 "[48,128]" 0.526 0.36 "[0.2]" 80 &












# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:4 '[10,30,50]'  0.5 0.5 '[-1]' 50 5 0.01 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:5 '[10,30,50]'  0.5 0.5 '[-1]' 30 5 0.01 &

# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:7 '[10,30,50]'  0.5 0.5 '[-1]' 50 1 0.01 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:5 '[10,30,50]'  0.5 0.5 '[-1]' 30 1 0.01 &

# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:4 '[10,30,50]'  0.5 0.5 '[-1]' 50 3 0.01 &
# wait
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:5 '[10,30,50]'  0.5 0.5 '[-1]' 30 3 0.001 &

# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:7 '[10,30,50]'  0.5 0.5 '[-1]' 50 5 0.001 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:4 '[10,30,50]'  0.5 0.5 '[-1]' 30 5 0.001 &

# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:7 '[10,30,50]'  0.5 0.5 '[-1]' 50 1 0.001 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:5 '[10,30,50]'  0.5 0.5 '[-1]' 30 1 0.001 &

wait
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:4 '[10,30,50]'  0.5 0.5 '[-1]' 50 3 0.1 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:5 '[10,30,50]'  0.5 0.5 '[-1]' 30 3 0.1 &

# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:3 '[10,30,50]'  0.5 0.5 '[-1]' 50 5 0.1 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:5 '[10,30,50]'  0.5 0.5 '[-1]' 30 5 0.1 &

# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:7 '[10,30,50]'  0.5 0.5 '[-1]' 50 1 0.1 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:3 '[10,30,50]'  0.5 0.5 '[-1]' 30 1 0.1 &




# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:4 '[8,16,32,64]'  0.5 0.5 '[-1]' 20 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:3 '[10,20,30,40]'  0.5 0.5 '[-1]' 20 &
wait
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:7 '[64,128,256,512]'  0.5 0.5 '[-1]' 256 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:5 '[8,16,32,64,128]'  0.5 0.5 '[-1]' 30 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:6 '[10,20,30,40,50]'  0.5 0.5 '[-1]' 30 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:4 '[64,128,256,512,1024]'  0.5 0.5 '[-1]' 256 &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:2 '[128,256]' 0.5 0.5 '[-1]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:3 '[4,8,16,32]' 0.5 0.5 '[-1]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:4 '[64,256,512,1024]' 0.5 0.5 '[-1]' &

# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:5 '[128,256,512,1024]' 0.5 0.5 '[-1]' &
# python kge/cli_debug.py models/fb15k-237/AdaE_auto.yaml cuda:6 '[64,128,256,512]' 0.5 0.5 '[-1]' &