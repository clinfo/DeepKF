dmm --config config_adw.json --hyperparam hyparam_adw.json --save-config config.result.json train,test,filter --gpu 0
dmm-plot train --config config.result.json --limit_all 10 
dmm-plot filter --config config.result.json --limit_all 10


