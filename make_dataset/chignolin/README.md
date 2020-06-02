# 準備
はじめにmdtrajとmsmbuilderをインストールします   
mdtraj (http://mdtraj.org/1.9.3/installation.html)   
msmbuilder(http://msmbuilder.org/3.8.0/installation.html) 

- mdtrajのインストールについて
```
conda install -c conda-forge mdtraj
```
- msmbuilderのインストールについて
conda, pip経由ではうまくいかないため, ソースをダウンロードしてインストールします.
```
git clone https://github.com/msmbuilder/msmbuilder.git
cd msmbuilder/
python setup.py install
```
msmbuilderはfastclusterに依存するため, これもインストールします.
```
conda install -c conda-forge fastcluster
```
下記コマンドでmsmbuilderが正しくインストールされたか確認してください.
```
msmb -h
```

# データ作成方法 
- 下記の要領で初期構造, トラジェクトリを指定して`make_data_contact.py`を実行
```
python make_data_contact.py
```

## 初期構造の指定
スクリプト内20行目   

> 20 list=["0_31410"]

で初期構造を指定します   
(例えば、0_31410は、traj_data/chig_data/traj_n1/0_31410を指します）      
traj_n1はD(Asp3N-Gly7O)D(Asp3N-Thr8O)のプロットの   
①: D(Asp3N-Thr8O)>1.5 [nm] & D(Asp3N-Gly7O)>1.5 [nm]    
を満たす領域から選んだ初期構造   

下記を参照ください      
<img width="606" alt="スクリーンショット 2020-03-08 18 32 48" src="https://user-images.githubusercontent.com/39581094/76160344-b177c480-616c-11ea-9054-ddb7e2d7f53f.png">

## トラジェクトリの指定

スクリプト内37行目

> 37     traj=name+'/protein_gpu/equil_n1/md1_noSOL_fit_skip10.xtc'   
でトラジェクトリ を指定します   

traj_data/chig_data/traj_n??/0_????/protein_gpu下   

- equil_n1 : 1 step= 2psで保存したトラジェクトリ    
- equil_n2 : 1 step= 2fsで保存したトラジェクトリ   

traj_data/chig_data/traj_n??/0_????/protein_gpu/equil_n1下
- md1_noSOL_fit.xtc             : 1 step= 2psで保存したトラジェクトリ 
- md1_noSOL_fit_skip2.xtc       : 1 step= 4psに間引いたトラジェクトリ 
- md1_noSOL_fit_skip5.xtc       : 1 step= 10psに間引いたトラジェクトリ 
- md1_noSOL_fit_skip10.xtc      : 1 step= 20psに間引いたトラジェクトリ

（トラジェクトリデータは、小島Google Driveの
```
traj_data/chig_data
```
から入手できます）      
