
export  CUDA_VISIBLE_DEVICES=3

mkdir -p sample_chig/model/sim
python dmm.py --config sample_chig/config.result.json --hyperparam sample_chig/model/hyparam.result.json field

python visualization/plot_chignolin_withpot.py --config ./sample_chig/config.result.json --hyperparam sample_chig/model/hyparam.result.json --limit_all 500 all
