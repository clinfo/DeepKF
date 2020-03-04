
export  CUDA_VISIBLE_DEVICES=3

python dmm.py --config sample_chig_gen/config_chig_gen.json --hyperparam sample_chig_gen/hyparam_chig_gen.json --save-config sample_chig_gen/config.result.json train,test,filter

python script/plot_p.py --config ./sample_chig_gen/config.result.json --hyperparam sample_chig_gen/model/hyparam.result.json --limit_all 10 all 
python script/plot.py --config ./sample_chig_gen/config.result.json --hyperparam sample_chig_gen/model/hyparam.result.json --limit_all 10 all

