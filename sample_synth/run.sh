cd `dirname $0`

rm -r model
mkdir -p model
mkdir -p model/result/
mkdir -p model/sim/

# data generator
python make.py

cd ..

export  CUDA_VISIBLE_DEVICES=3

python dmm.py --config sample_synth/config.json --hyperparam sample_synth/hyparam.json train --save-config ./sample_synth/config.result.json
python dmm.py --config sample_synth/config.result.json infer
python dmm.py --config ./sample_synth/config.result.json filter
python attractor.py --config ./sample_synth/config.result.json field

python script/plot_p.py --config ./sample_synth/config.result.json all 
python script/plot.py --config ./sample_synth/config.result.json all
python script/plot_vec.py ./sample_synth/config.result.json all

