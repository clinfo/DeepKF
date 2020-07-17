dmm train,test,filter --config config_adw_gen.json --hyperparam hyparam_adw_gen_005.json --save config_adw_gen_005.result.json --gpu 0
dmm-plot train --config config_adw_gen_005.result.json
dmm-plot filter --config config_adw_gen_005.result.json 