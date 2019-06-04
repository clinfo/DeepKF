cd `dirname $0`
cd ..

echo "... generating configs"
mkdir -p sample_cycle/experiments/configs/
sh sample_cycle/experiments/gen_configs.sh


export  CUDA_VISIBLE_DEVICES=3
cfg=sample_cycle/config.json
for f in `ls sample_cycle/experiments/configs/config.*.json`
do
b=`basename ${f} .json`
c=${b#config.}
echo "...running:" $c
h=sample_cycle/experiments/model.${c}/hyparam.result.json

python dmm.py --config ${cfg} --hyperparam ${f} --save-config ${h} train,infer,filter,field


python script/plot_p.py --config ${h} all 
python script/plot.py --config ${h} all
python script/plot_vec.py ${h} all

done
