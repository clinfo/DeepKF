

python dkf.py --config config_train_2d.json --hyperparam ./model_2d/hyparam.result.json train

hyfile="./model_2d/hyparam.result.json"


python dkf.py --config hyopt/config_infer.json --hyperparam ${hyfile} --save-config ./model_2d/config_infer2.json infer

python dkf.py --config config_train_2d.json --hyperparam model_2d/hyparam.result.json filter
python attractor.py --config config_train_2d.json --hyperparam model_2d/hyparam.result.json field
python attractor.py --config config_train_2d.json --hyperparam model_2d/hyparam.result.json infer

python plot_p.py ./model_2d/config_infer2.json 
python plot.py model_2d/config_infer2.json 100
python plot_vec.py model_2d/config_infer2.json

