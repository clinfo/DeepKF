alpha=$1
beta=$2
seed=$3
dmm filter --config config_generate.json --hyperparam hyperparams/${alpha}_${beta}_${seed}.json
python ../script/plot_filter.py results/${alpha}_${beta}_${seed}/filter.jbl results/${alpha}_${beta}_${seed}/generate.png 
