cd `dirname $0`
cd ..

export  CUDA_VISIBLE_DEVICES=2

python dmm.py --config sample_segmentation/config.json --hyperparam sample_segmentation/hyparam.json --save-config ./sample_segmentation/config.result.json \
	train,infer,filter,field
python dmm.py --config ./sample_segmentation/config.result.json --hyperparam sample_segmentation/hyparam.json filter_discrete

python script/plot_forward.py --config ./sample_segmentation/config.result.json --hyperparam sample_segmentation/hyparam.json all
#python script/plot_s.py --config ./sample_segmentation/config.result.json --hyperparam sample_segmentation/hyparam.json all 
#python script/plot_sp.py --config ./sample_segmentation/config.result.json --hyperparam sample_segmentation/hyparam.json all 
#python script/plot.py --config ./sample_segmentation/config.result.json --hyperparam sample_segmentation/model/hyparam.result.json all
#python script/plot_vec.py ./sample_segmentation/config.result.json all

