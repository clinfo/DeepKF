### ***アラニンジペプチドトラジェクトリ を用いたDKFを行う***

![ala](https://user-images.githubusercontent.com/39581094/76176955-f3008200-61f5-11ea-9063-6e87532dccd6.png)

アラニンジペプチドでは、特徴量として二面角（φ・ψ）を抽出します  
入力データの作成は、
```
make_dataset/alanine/make_data_ala.py
```
を参照ください   
作成されたデータは、小島Google Driveの   
```
traj_data/input_files/ala_traj_all.npy
```
およびpe1計算機の
```
/data/traj_data/input_files/ala_traj_all.npy
```
にあります   
ala_traj_all.npyの形状は、(600, 300, 2)としています （# 変更可能）

DeepKF/sample_ala/dataset 下に解析するトラジェクトリデータ (今回は、ala_traj_all.npy)を置きます   
pe1計算機上では, 当該フォルダにて下記コマンドを実行し, トラジェクトリデータへのシンボリックリンクを貼ってください.
```
ln -s /data/traj_data/input_files/ala_traj_all.npy
```

sample_ala下に、config.josnとhyparam.jsonを置いています  
（今回の設定例は、  
```
DeepKF/setting_examples/config_ala.json   
DeepKF/setting_examples/hyparam_ala.json  
```
にあります    
    

**実行コマンド**   
下記コマンドでDKFを実行します   
   
```
cd DeepKF  
mkdir -p sample_ala/model  
sh sample_adw/run_ala.sh
```
さらに、下記コマンドで二面角情報上に、dimの値で色付けしたプロットを作成します   

```
sh sample_ala/plot_ala.sh
```

アラニンジペプチドには3つのメジャー領域が知られています（下図 αR、αL、β）   
<img width="759" alt="ala_state" src="https://user-images.githubusercontent.com/39581094/76176962-f85dcc80-61f5-11ea-8368-fac4db55eb03.png">

この3領域を区別できることを確認します   
![ala_dkf](https://user-images.githubusercontent.com/39581094/76176965-fc89ea00-61f5-11ea-9d3d-c53aaa54c777.png)



