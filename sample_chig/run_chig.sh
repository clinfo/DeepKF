
export  CUDA_VISIBLE_DEVICES=3

python dmm.py --config sample_chig/config_chig.json --hyperparam sample_chig/hyparam_chig.json --save-config sample_chig/config.result.json train,test,filter

python script/plot_p.py --config ./sample_chig/config.result.json --hyperparam sample_chig/model/hyparam.result.json --limit_all 10 --num_dim 1 --num_particle 1 all 
python script/plot.py --config ./sample_chig/config.result.json --hyperparam sample_chig/model/hyparam.result.json --limit_all 10 all

