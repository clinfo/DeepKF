
export  CUDA_VISIBLE_DEVICES=3

python dmm.py --config sample_br2d/config_br2d.json --hyperparam sample_br2d/hyparam_br2d.json --save-config sample_br2d/config.result.json train,test,filter

python script/plot_p.py --config ./sample_br2d/config.result.json --hyperparam sample_br2d/model/hyparam.result.json --limit_all 10 all 
python script/plot.py --config ./sample_br2d/config.result.json --hyperparam sample_br2d/model/hyparam.result.json --limit_all 10 all

