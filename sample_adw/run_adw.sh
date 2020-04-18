
export  CUDA_VISIBLE_DEVICES=3

python dmm.py --config sample_adw/config_adw.json --hyperparam sample_adw/hyparam_adw.json --save-config sample_adw/config.result.json train,test,filter

python script/plot_p.py --config ./sample_adw/config.result.json --hyperparam sample_adw/model/hyparam.result.json --limit_all 10 all 
python script/plot.py --config ./sample_adw/config.result.json --hyperparam sample_adw/model/hyparam.result.json --limit_all 10 all

