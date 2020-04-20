### ***シニョリンのトラジェクトリ を用いたDKFを行う***


シニョリンでは、特徴量として重原子間距離を抽出します   
入力データの作成は、   
```
make_dataset/chignolin/make_data_contact.py
```
を参照ください   
   
作成されたデータは、小島Google Driveの   
```
traj_data/input_files/chig_dkf_train.npy
traj_data/input_files/chig_dkf_test.npy
```
およびpe1計算機の
```
/data/traj_data/input_files/chig_dkf_train.npy
/data/traj_data/input_files/chig_dkf_test.npy
```
にあります

DeepKF/sample_chig/dataset 下に解析するトラジェクトリデータ (今回は、chig_dkf_train.npy & chig_dkf_test.npy)を置きます   
pe1計算機上では, 当該フォルダにて下記コマンドを実行し, トラジェクトリデータへのシンボリックリンクを貼ってください.
```
ln -s /data/traj_data/input_files/chig_dkf_train.npy
ln -s /data/traj_data/input_files/chig_dkf_test.npy
```

chig_dkf_train.npyは、1step=2psの9マイクロ秒のトラジェクトリデータ   
chig_dkf_test.npyは、1step=2psの1マイクロ秒のトラジェクトリデータ   


sample_chig下に、config.josnとhyparam.jsonを置いています  
（今回の設定例は、  
```
DeepKF/setting_examples/config_chig.json   
DeepKF/setting_examples/hyparam_chig.json  
```
にあります   


**実行コマンド**   
下記コマンドでDKFを実行します   
   
```
cd DeepKF  
mkdir -p sample_chig/model  
sh sample_chig/run_chig.sh
```
下記コマンドでD(Asp3N-Gly7O)とD(Asp3N-Thr8O)上に、dimの値で色付けしたプロットを作成します  
```
sh sample_chig/plot_chig.sh
```
下図Dimの色が青になっている領域（Native領域）を区別していることを確認します   
![chig_dkf](https://user-images.githubusercontent.com/39581094/76176594-c13aeb80-61f4-11ea-8b1e-5a8a733e7ce0.png)

（参考）下図は温度一定MDでのプロット   
<img width="471" alt="chig_state" src="https://user-images.githubusercontent.com/39581094/76176605-c6983600-61f4-11ea-98b3-446167eb0fdd.png">


さらに計算した状態空間を用いて擬似ポテンシャルを作成します   
```
sh sample_chig/field.sh
```
Dimの色が青になっている領域（Native領域）が安定であることを確認します   

![chig_dkf_withpot](https://user-images.githubusercontent.com/39581094/76176625-cf890780-61f4-11ea-9978-7e4a1607ac71.png)

