alpha=${1}
beta=${2}
seed=${3}
<<<<<<< HEAD
#mkdir -p results/${alpha}_${beta}_${seed}/
#dmm train,test --config config_learn.json --hyperparam hyperparams/${alpha}_${beta}_${seed}.json\
# --save-config config_results/${alpha}_${beta}_${seed}.json --seed ${seed} --gpu $((seed % 3 + 1))
#dmm-plot infer --config config_results/${alpha}_${beta}_${seed}.json --limit_all 5
#python ../script/plot_loss.py results/${alpha}_${beta}_${seed}/log.txt results/${alpha}_${beta}_${seed}/plot/loss.png
cp results/${alpha}_${beta}_${seed}/log.txt results/${alpha}_${beta}_${seed}/learn_log.tsv
=======
mkdir -p results/${alpha}_${beta}_${seed}/
dmm train,test --config config_learn.json --hyperparam hyperparams/${alpha}_${beta}_${seed}.json\
 --save-config config_results/${alpha}_${beta}_${seed}.json --seed ${seed} --gpu $((seed % 3 + 1))
dmm-plot infer --config config_results/${alpha}_${beta}_${seed}.json --limit_all 5
python ../script/plot_loss.py results/${alpha}_${beta}_${seed}/log.txt results/${alpha}_${beta}_${seed}/plot/loss.png
cp results/${alpha}_${beta}_${seed}/log.txt results/${alpha}_${beta}_${seed}/learn_log.txt
>>>>>>> b10033e7d8a1d37e2430a767f5eac204f2cb2038
