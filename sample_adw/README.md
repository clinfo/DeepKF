***ブラウン運動トラジェクトリ を使ってDKFを行う***

**DeepKF/sample_adw/dataset** 下に解析するトラジェクトリデータ (今回は、adw_traj1.npy)を置きます  
  
（adw_traj1.npyは 
```
DeepKF/make_dataset/brown_model/make_data_adw.py 
``` 
で作成されます。）　　

sample_adw下に、config.josnとhyparam.jsonを置いています  
（今回の設定例は、  
```
DeepKF/setting_examples/config_adw.json   
DeepKF/setting_examples/hyparam_adw.json  
```
にあります）  

config_adw.json内の  
```
"data_test_npy": "sample_adw/dataset/adw_traj1.npy",  
"data_train_npy": "sample_adw/dataset/adw_traj1.npy"  
```
にトラジェクトリデータの場所を指定します  

**実行コマンド**   
下記のコマンドでDKFを実行します   
   
```
cd DeepKF  
mkdir -p sample_adw/model  
sh sample_adw/run_adw.sh
```


  
