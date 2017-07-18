python hyopt_gen.py

./hyopt/run,sh

python hyopt_result.py > hyopt/result.txt
hyfile=`cat hyopt/result.txt | grep "## Top" | cut -d " " -f 3`

python dkf.py --config hyopt/config_infer.json --hyperparam ${hyfile} infer

python dkf.py --config hyopt/config_infer.json --hyperparam ${hyfile} --save-config ./hyopt/config_infer2.json infer

