mkdir -p sample_synth/experiments
mkdir -p sample_synth/experiments/configs/

export  CUDA_VISIBLE_DEVICES=3
cfg=sample_synth/config.json
for f in `ls sample_synth/experiments/configs/config.*.json`
do
b=`basename ${f} .json`
c=${b#config.}
echo $c
h=sample_synth/experiments/model.${c}/hyparam.result.json
python dmm.py --config ${cfg} --hyperparam ${f} train 
#c=sample_synth/experiments/model${d}/
python dmm.py --config ${cfg} --hyperparam ${f} --save-config ${h} infer
python dmm.py --config ${cfg} --hyperparam ${h} filter
python attractor.py --config ${cfg} --hyperparam ${h} field

python script/plot_p.py --config ${cfg} --hyperparam ${h} all 
python script/plot.py --config ${cfg} --hyperparam ${h} all
python script/plot_vec.py ${h} all

done
