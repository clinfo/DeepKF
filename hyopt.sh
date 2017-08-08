python hyopt_gen.py

./hyopt/run,sh

python hyopt_result.py > hyopt/result.txt
hyfile=`cat hyopt/result.txt | grep "## Top" | cut -d " " -f 3`

python dkf.py --config hyopt/config_infer.json --hyperparam ${hyfile} infer

python dkf.py --config hyopt/config_infer.json --hyperparam ${hyfile} --save-config ./hyopt/config_infer2.json infer

#nohup python dkf.py --config config_train_2d.json --hyperparam ./model_2d/config_infer2.json train > model_2d/log.txt
# python dkf.py --config ./config_train_2d.json --hyperparam ./model_2d/hyparam.result.json --save-config ./model_2d/config_infer2.json infer
python plot.py model_2d/config_infer2.json 100
python dkf.py --config config_train_2d.json --hyperparam model_2d/hyparam.result.json filter
python plot_p.py ./model_2d/config_infer2.json 
python attractor.py --config config_train_2d.json --hyperparam model_2d/hyparam.result.json field
python attractor.py --config config_train_2d.json --hyperparam model_2d/hyparam.result.json infer
python plot_vec.py model_2d/config_infer2.json
