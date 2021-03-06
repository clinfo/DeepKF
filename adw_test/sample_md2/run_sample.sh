
export  CUDA_VISIBLE_DEVICES=3

python dmm.py --config sample_md2/config_sample.json --hyperparam sample_md2/hyparam_sample.json --save-config ./sample_md2/config.result.json train,test,filter

python script/plot_p.py --config ./sample_md2/config.result.json --hyperparam sample_md2/model/hyparam.result.json --limit_all 10 all 
python script/plot.py --config ./sample_md2/config.result.json --hyperparam sample_md2/model/hyparam.result.json --limit_all 10 all
