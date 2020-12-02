
export  CUDA_VISIBLE_DEVICES=3

#python dmm.py --config sample_ala/config_ala.json --hyperparam sample_ala/hyparam_ala.json --save-config sample_ala/config.result.json train,test,filter
dmm --config sample_ala/config_ala.json --hyperparam sample_ala/hyparam_ala.json --save-config sample_ala/config.result.json train,test,filter

#python script/plot_p.py --config ./sample_ala/config.result.json --hyperparam sample_ala/model/hyparam.result.json --limit_all 10 --num_dim 1 all 
#python script/plot.py --config ./sample_ala/config.result.json --hyperparam sample_ala/model/hyparam.result.json --limit_all 10 all

