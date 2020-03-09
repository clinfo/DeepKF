cd `dirname $0`

# data generator
#python make.py

dmm train,infer,filter,field --config config_base.json --hyperparam hyparam_base.json
dmm-plot infer --config config_base.json
dmm-plot filter --config config_base.json
dmm-plot infer --config config_base.json  --z_plot_type scatter --anim
dmm-field-plot infer --config config_base.json

