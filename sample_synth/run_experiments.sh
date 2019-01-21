cd `dirname $0`
cd ..

mkdir -p sample_synth/experiments/configs/
sh sample_synth/experiments/gen_configs.sh


export  CUDA_VISIBLE_DEVICES=3
cfg=sample_synth/config.json
for f in `ls sample_synth/experiments/configs/config.*.json`
do
b=`basename ${f} .json`
c=${b#config.}
echo $c
h=sample_synth/experiments/model.${c}/hyparam.result.json

python dmm.py --config ${cfg} --hyperparam ${f} --save-config ${h} train,infer,filter,field


python script/plot_p.py --config ${h} all 
python script/plot.py --config ${h} all
python script/plot_vec.py ${h} all

done
