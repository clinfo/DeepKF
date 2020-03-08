***シニョリンのトラジェクトリ を用いたDKFを行う***


シニョリンでは、特徴量として重原子間距離を抽出します   
入力データの作成は、   
```
make_dataset/alanine/make_data_ala.py
```
を参照ください   
   
作成されたデータは、小島Google Driveの   
```
traj_data/input_files/chig_dkf_train.npy
traj_data/input_files/chig_dkf_test.npy
```
にあります   

DeepKF/sample_chig/dataset 下に解析するトラジェクトリデータ (今回は、chig_dkf_train.npy & chig_dkf_test.npy)を置きます   

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
![500_data499_plot](https://user-images.githubusercontent.com/39581094/76159978-63ad8d00-6169-11ea-969c-a83597c5b66e.png)

（参考）下図は温度一定MDでのプロット   
<img width="471" alt="スクリーンショット 2020-03-08 18 17 07" src="https://user-images.githubusercontent.com/39581094/76159988-6c9e5e80-6169-11ea-9e9e-31b18f990108.png">


さらに計算した状態空間を用いて擬似ポテンシャルを作成します   
```
sh sample_chig/field.sh
```
Dimの色が青になっている領域（Native領域）が安定であることを確認します   

![test_plot](https://user-images.githubusercontent.com/39581094/76159983-67d9aa80-6169-11ea-90b0-bdf68f0d177f.png)

