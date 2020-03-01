***ブラウン運動トラジェクトリ を使ってDKFを行う***

**DeepKF/sample_md2/dataset** 下に解析するトラジェクトリデータ (今回は、adw_traj1.npy)を置きます  
  
（adw_traj1.npyは 
```
DeepKF/make_dataset/brown_model/make_data_adw.py 
``` 
で作成されます。）　　

sample_md2下に、config.josnとhyparam.jsonを置いています。  
（今回の設定例は、  
```
DeepKF/setting_examples/config_adw.json   
DeepKF/setting_examples/hyparam_adw.json  
```
にあります）  

config_adw.json内の  
```
"data_test_npy": "sample_md2/dataset/adw_traj1.npy",  
"data_train_npy": "sample_md2/dataset/adw_traj1.npy"  
```
にトラジェクトリデータの場所を指定します  

**実行コマンド**   
下記のコマンドでDKFを実行します   
   
```
cd DeepKF  
mkdir -p sample_md2/model  
sh sample_md2/run_sample.sh
```
  
