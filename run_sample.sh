
export  CUDA_VISIBLE_DEVICES=0

python dmm.py --config sample/config.json --hyperparam sample/hyparam.json --save-config ./model/config.result.json --seed 1 train,test,filter,field

python script/plot_p.py --config model/config.result.json --hyperparam model/hyparam.result.json all 
python script/plot.py --config model/config.result.json --hyperparam model/hyparam.result.json all
python script/plot_vec.py model/config.result.json all

