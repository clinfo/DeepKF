
export  CUDA_VISIBLE_DEVICES=3

python visualization/plot_chignolin.py --config ./sample_chig/config.result.json --hyperparam sample_chig/model/hyparam.result.json --limit_all 500 all

