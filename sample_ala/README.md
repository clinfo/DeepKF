***アラニンジペプチドトラジェクトリ を用いたDKFを行う***

![ala](https://user-images.githubusercontent.com/39581094/75623684-e2497e00-5bef-11ea-98b8-3708ad8f72cd.png)

アラニンジペプチドでは、特徴量として二面角（φ・ψ）を抽出します  
入力データの作成は、
```
make_dataset/alanine/make_data_ala.py
```
を参照ください   
作成されたデータは、Google Driveの   
```
traj_data/input_files/ala_traj_all.npy
```
にあります   
ala_traj_all.npyの形状は、(600, 300, 2)としています （# 変更可能）

DeepKF/sample_adw/dataset 下に解析するトラジェクトリデータ (今回は、ala_traj_all.npy)を置きます   

**実行コマンド**   
下記コマンドでDKFを実行します   
   
```
cd DeepKF  
mkdir -p sample_ala/model  
sh sample_adw/run_ala.sh
```
さらに、下記コマンドで二面角の情報にq、dimの値で色付けしたプロットを作成します   

```
sh sample_adw/plot_ala.sh
```


