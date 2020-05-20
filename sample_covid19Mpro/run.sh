


dmm train --config config.json --hyperparam hyparam_base.json
dmm infer,filter,field --config config.json --hyperparam hyparam_base.json
dmm-plot infer --config config.json --limit_all 100 --x_plot_type scatter
