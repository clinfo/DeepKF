***下記ポテンシャル（foldingモデル）上を動く2D ブラウン運動トラジェクトリ を使ってDKFを行う***


**DeepKF/sample_br2d/dataset** 下に解析するトラジェクトリデータ (今回は、folding_2d_traj.npy)を置きます  
  
（folding_2d_traj.npyは 
```
DeepKF/make_dataset/brown_model/make_data_brown.py 
``` 
で作成されます。）　　

sample_br2d下に、config.josnとhyparam.jsonを置いています  
（今回の設定例は、  
```
DeepKF/setting_examples/config_br2d.json   
DeepKF/setting_examples/hyparam_br2d.json  
```
にあります）  

config_adw.json内の  
```
"data_test_npy": "sample_br2d/dataset/folding_2d_traj.npy",  
"data_train_npy": "sample_br2d/dataset/folding_2d_traj.npy"  
```
にトラジェクトリデータの場所を指定します  

**実行コマンド**   
下記のコマンドでDKFを実行します   
   
```
cd DeepKF  
mkdir -p sample_br2d/model  
sh sample_adw/run_br2d.sh
```
