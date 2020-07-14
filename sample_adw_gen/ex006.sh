dmm train,test,filter --config config_adw_gen.json --hyperparam hyparam_adw_gen_006.json --save config_adw_gen_006.result.json --gpu 0
dmm-plot train --config config_adw_gen_006.result.json
dmm-plot filter --config config_adw_gen_006.result.json 