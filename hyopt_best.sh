id=`python hyopt_result.py | grep "^## Top_ID" | cut -d" " -f3`
cfg=hyopt/result${id}/config.result.json
python dkf.py --config hyopt/config_train.json --hyperparam ./hyopt/hyparam${id}.result.json --save-config ${cfg} infer
python dkf.py --config ${cfg} --hyperparam ./hyopt/hyparam${id}.result.json  filter
mkdir hyopt/result${id}/sim
mkdir hyopt/result${id}/plot
python attractor.py --config ${cfg} --hyperparam ./hyopt/hyparam${id}.result.json field
python script/plot_vec.py ${cfg} all

