dmm train,test,filter --config config_adw_gen.json --hyperparam hyparam_adw_gen_002.json --save config_adw_gen_002.result.json --gpu 2
dmm-plot train --config config_adw_gen_002.result.json
dmm-plot filter --config config_adw_gen_002.result.json 