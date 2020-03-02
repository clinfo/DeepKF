***下記のポテンシャル上（assymetry double well potential）を動く1D ブラウン運動トラジェクトリ を使ってDKFを行う***

![adw_potential](https://user-images.githubusercontent.com/39581094/75623493-ce048180-5bed-11ea-87ce-79103efbc7cf.png)  

DeepKF/sample_adw/dataset 下に解析するトラジェクトリデータ (今回は、adw_traj1.npy)を置きます  
  
（adw_traj1.npyは 
```
DeepKF/make_dataset/brown_model/make_data_adw.py 
``` 
で作成されます。   
このデータは100step分の1次元の座標データを500本分としています↓   
```
import numpy as np
traj=np.load("adw_traj1.npy")
traj.shape #(500, 100, 1)
```
）　

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
下記コマンドでDKFを実行します   
   
```
cd DeepKF  
mkdir -p sample_adw/model  
sh sample_adw/run_adw.sh
```

plot_adw_test下に実行結果のPNGファイルを置いています


![adw](https://user-images.githubusercontent.com/39581094/75623342-66016b80-5bec-11ea-87a6-cab205fefd56.png)
  

***※上手くいかない場合***
```
DeepKF/adw_test
```
をご参照ください  
sample_md2が実行済みフォルダです   
下記コマンドで結果を確認してみてください
```
cd adw_test
cp -r sample_md2 sample_md2_old        #実行済みの結果をsample_md2_oldに移す
rm -r sample_md2/model                 #sample_md2下の実行結果を消す
rm -r sample_md2/config.result.json　　 #sample_md2下の実行結果を消す
mkdir -p sample_md2/model  
sh sample_md2/run_sample.sh            #新たに実行
```
