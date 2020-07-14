dmm train,test,filter --config config_adw_gen.json --hyperparam hyparam_adw_gen_001.json --save config_adw_gen_001.result.json --gpu 1
dmm-plot train --config config_adw_gen_001.result.json
dmm-plot filter --config config_adw_gen_001.result.json 