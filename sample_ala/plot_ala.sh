
export  CUDA_VISIBLE_DEVICES=3

python visulalization/plot_ala_phi_psi_cl_heatmap.py --config ./sample_ala/config.result.json --hyperparam sample_ala/model/hyparam.result.json --limit_all 600 all

