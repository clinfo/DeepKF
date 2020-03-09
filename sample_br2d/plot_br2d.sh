
export  CUDA_VISIBLE_DEVICES=3

python visualization/plot_brown2d_heatmap.py --config ./sample_br2d/config.result.json --hyperparam sample_br2d/model/hyparam.result.json --limit_all 1000 all

