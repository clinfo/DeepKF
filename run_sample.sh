
export  CUDA_VISIBLE_DEVICES=0

python dkf.py --config sample/config.json --hyperparam sample/hyparam.json train 
python dkf.py --config sample/config.json --hyperparam model/hyparam.result.json --save-config ./model/config.result.json infer
python dkf.py --config model/config.result.json --hyperparam model/hyparam.result.json filter
python attractor.py --config model/config.result.json --hyperparam model/hyparam.result.json field

python script/plot_p.py --config model/config.result.json --hyperparam model/hyparam.result.json all 
python script/plot.py --config model/config.result.json --hyperparam model/hyparam.result.json all
python script/plot_vec.py model/config.result.json all

