cd `dirname $0`

# data generator
rm -r model
mkdir -p model
mkdir -p model/result/
mkdir -p model/sim/

python make.py

cd ..

export  CUDA_VISIBLE_DEVICES=3

python dmm.py --config sample_cycle/config.json --hyperparam sample_cycle/hyparam.json --save-config ./sample_cycle/config.result.json train,infer,filter,field
#python dmm.py --config ./sample_cycle/config.result.json filter,field

python script/plot_p.py --config ./sample_cycle/config.result.json all 
python script/plot.py --config ./sample_cycle/config.result.json all
python script/plot_vec.py ./sample_cycle/config.result.json all

