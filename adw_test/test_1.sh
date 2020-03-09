
#hyfile="./model1/hyparam.result.json"
export CUDA_VISIBLE_DEVICES=1

python3 dkf.py --config config_train_small.json --hyperparam ./model1/hyparam.json train > model1/log.txt


python3 dkf.py --config config_train_small.json --hyperparam ./model1/hyparam.result.json --save-config ./model1/config_infer.json infer

python3 dkf.py --config config_train_small.json --hyperparam model1/hyparam.result.json filter

python3 attractor.py --config config_train_small.json --hyperparam model1/hyparam.result.json field
python3 attractor.py --config config_train_small.json --hyperparam model1/hyparam.result.json infer
python3 attractor.py --config config_train_small.json --hyperparam model1/hyparam.result.json potential

python3 script/plot_p.py ./model1/config_infer.json all 
python3 script/plot.py model1/config_infer.json all
python3 script/plot_vec.py model1/config_infer.json all
python3 script/plot_pot.py model1/config_infer.json all

