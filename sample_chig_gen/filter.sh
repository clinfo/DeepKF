
export  CUDA_VISIBLE_DEVICES=3

python dmm.py --config sample_chig_gen/config.result.json --hyperparam sample_chig_gen/hyparam_chig_gen.json --save-config sample_chig_gen/config.result.json filter

python script/plot_p.py --config ./sample_chig_gen/config.result.json --hyperparam sample_chig_gen/model/hyparam.result.json --limit_all 10 --num_dim 1 --num_particle 1 all

