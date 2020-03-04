
export  CUDA_VISIBLE_DEVICES=3

python script/plot_p.py --config ./sample_ala/config.result.json --hyperparam sample_ala/model/hyparam.result.json --limit_all 10 --num_dim 1 all 

