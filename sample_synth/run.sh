cd `dirname $0`
cd ..

export  CUDA_VISIBLE_DEVICES=3

python dmm.py --config sample_synth/config.json --hyperparam sample_synth/hyparam.json train 
python dmm.py --config sample_synth/config.json --hyperparam sample_synth/model/hyparam.result.json --save-config ./sample_synth/config.result.json infer
python dmm.py --config ./sample_synth/config.result.json --hyperparam sample_synth/model/hyparam.result.json filter
python attractor.py --config ./sample_synth/config.result.json --hyperparam sample_synth/model/hyparam.result.json field

python script/plot_p.py --config ./sample_synth/config.result.json --hyperparam sample_synth/model/hyparam.result.json all 
python script/plot.py --config ./sample_synth/config.result.json --hyperparam sample_synth/model/hyparam.result.json all
python script/plot_vec.py ./sample_synth/config.result.json all

