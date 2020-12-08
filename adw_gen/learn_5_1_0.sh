dmm train,test --config config_train_and_test.json --hyperparam hyparam/5_1_0.json --save config_train_and_test.result.json --gpu 0
dmm-plot infer --config config_train_and_test.results.json
python ../script/plot_loss.py results/result_5_1/ex0/log.txt results/result_5_1/ex0/plot/loss.png
cp results/result_5_1/ex0/log.txt results/result_5_1/ex0/learn_log.txt
