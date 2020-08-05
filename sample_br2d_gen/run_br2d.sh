
dmm train,test,filter --config config_br2d.json --hyperparam hyparam_br2d.json --save-config config.result.json --gpu 0

#python script/plot_p.py --config ./sample_br2d/config.result.json --hyperparam sample_br2d/model/hyparam.result.json --limit_all 10 all 
#python script/plot.py --config ./sample_br2d/config.result.json --hyperparam sample_br2d/model/hyparam.result.json --limit_all 10 all

