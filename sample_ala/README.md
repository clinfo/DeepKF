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
sh sample_adw/plot_ala.sh
```

アラニンジペプチドには3つのメジャー領域が知られています（下図 αR、αL、β）   
<img width="759" alt="ala_state" src="https://user-images.githubusercontent.com/39581094/75691429-82360300-5ce7-11ea-88bd-5257a1a8e174.png">   

この3領域を区別できることを確認します   
![alanine](https://user-images.githubusercontent.com/39581094/75691605-a2fe5880-5ce7-11ea-9aa6-f22137da55f2.png)



