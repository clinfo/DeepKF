dmm train,test,filter --config config_adw_gen.json\
    --hyperparam hyparam_adw_gen_vae.json\
    --save-config config_adw_gen_vae.result.json --gpu 0
dmm-plot infer --config config_adw_gen_vae.result.json --limit_all 5
dmm-plot filter --config config_adw_gen_vae.result.json --limit_all 5