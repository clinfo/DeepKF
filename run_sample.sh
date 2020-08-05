
#export  CUDA_VISIBLE_DEVICES=0

#python dmm.py --config sample/config.json --hyperparam sample/hyparam.json --save-config ./model/config.result.json train,test,filter,field
dmm train,test,filter,field --config sample/config.json\
 --hyperparam sample/hyparam.json\
 --save-config model/config.result.json --gpu 0

#python script/plot_p.py --config model/config.result.json --hyperparam model/hyparam.result.json all 
#python script/plot.py --config model/config.result.json --hyperparam model/hyparam.result.json all
dmm-plot infer --config model/config.result.json --hyperparam model/hyparam.result.json 
#python script/plot_vec.py model/config.result.json all

