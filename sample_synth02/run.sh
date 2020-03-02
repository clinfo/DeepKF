cd `dirname $0`

# data generator
python make.py

dmm train,infer --config config.json --hyperparam hyparam.json --save-config ./config.result.json
#dmm train,infer,filter,field --config config.json --hyperparam hyparam.json --save-config ./config.result.json

#python script/plot_p.py --config ./sample_synth/config.result.json all 
dmm-plot infer --config config.json
#dmm-plot filter --config config.json
#python script/plot.py --config ./sample_synth/config.result.json all
#python script/plot_vec.py ./sample_synth/config.result.json all

#dmm-plot infer --config config.json --z_plot_type scatter --anim
