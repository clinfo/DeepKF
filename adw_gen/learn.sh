alpha=${1}
beta=${2}
seed=${3}
mkdir -p results/${alpha}_${beta}_${seed}/
dmm train,test --config config_learn.json --hyperparam hyperparams/${alpha}_${beta}_${seed}.json\
 --save-config config_results/${alpha}_${beta}_${seed}.json --seed ${seed} --gpu $((seed % 3 + 1))
dmm-plot infer --config config_results/${alpha}_${beta}_${seed}.json --limit_all 5
python ../script/plot_loss.py results/${alpha}_${beta}_${seed}/log.txt results/${alpha}_${beta}_${seed}/plot/loss.png
cp results/result_${alpha}_${beta}_${seed}/log.txt results/${alpha}_${beta}_${seed}/learn_log.txt
