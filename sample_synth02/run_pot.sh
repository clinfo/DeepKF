cd `dirname $0`

# data generator
#python make.py

dmm train,infer,filter,field --config config_pot.json --hyperparam hyparam_pot.json
dmm-plot infer --config config_pot.json
dmm-plot filter --config config_pot.json
dmm-plot infer --config config_pot.json  --z_plot_type scatter --anim
dmm-field-plot infer --config config_pot.json

