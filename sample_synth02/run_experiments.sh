cd `dirname $0`

# data generator
#python make.py
#python make_config.py


for f in `ls experiments/config/config_base*.json`
do
echo $f
dmm train,infer,filter,field --config ${f} --hyperparam hyparam_base.json
dmm-plot infer --config ${f}
dmm-plot filter --config ${f}
dmm-plot infer --config ${f}  --z_plot_type scatter --anim
dmm-field-plot infer --config ${f}
done

for f in `ls experiments/config/config_pot*.json`
do
echo $f
dmm train,infer,filter,field --config ${f} --hyperparam hyparam_pot.json
dmm-plot infer --config ${f}
dmm-plot filter --config ${f}
dmm-plot infer --config ${f}  --z_plot_type scatter --anim
dmm-field-plot infer --config ${f}
done

