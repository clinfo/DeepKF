
#hyfile="./model_2d/hyparam.result.json"

python dkf.py --config config_train.json --hyperparam ./model_2d/hyparam.json train > model_2d/log.txt


python dkf.py --config hyopt/config_infer.json --hyperparam ./model_2d/hyparam.result.json --save-config ./model_2d/config_infer2.json infer

python dkf.py --config config_train_2d.json --hyperparam model_2d/hyparam.result.json filter
python attractor.py --config config_train_2d.json --hyperparam model_2d/hyparam.result.json field
python attractor.py --config config_train_2d.json --hyperparam model_2d/hyparam.result.json infer
python attractor.py --config config_train_2d.json --hyperparam model_2d/hyparam.result.json potential

python script/plot_p.py ./model_2d/config_infer2.json all 
python script/plot.py model_2d/config_infer2.json all
python script/plot_vec.py model_2d/config_infer2.json

python script/plot_pot.py model_2d/config_infer2.json
