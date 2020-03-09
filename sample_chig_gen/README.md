### ***トラジェクトリ 生成（シニョリンを用いたDKF）***

**※※まだ検討段階（現時点で上手くいっていません）**

DeepKF/sample_chig/dataset　下に解析するトラジェクトリ データ（今回は、md2_train.npy & md2_test.npy）を置きます   

データは、　小島Google Driveの   
```
traj_data/input_files/md2_train.npy   
traj_data/input_files/md2_test.npy   
```
にあります    

md2_train.npyは、1step =2fs、9ns分のトラジェクトリ データ   
md2_test.npyは、1step =2fs、1ns分のトラジェクトリ データ   

このデータは
```
make_dataset/chignolin/make_data_contact.py
```
からも作成できます   

***実行コマンド***   
下記コマンドでDKFを実行します   

```
cd DeepKF   
mkdir -p sample_chig_gen/model   
sh sample_chig_gen/run_chig_gen.sh   
```

次に、config.result.jsonにマスクデータ（0・1データ）を指定します↓
> "mask_test.npy": sample_chig_gen/dataset/md2_mask.npy

このマスクデータは
```
python sample_chig_gen/dataset/maskdata.py
```
でも作成できます   
（マスクデータで0とした部分を、DKFで生成します）   

下記は潜在空間の次元 Dim=1, 3で検討した結果です   

二段目の青のプロットで欠けている部分が、マスクした部分です   
この部分をDKFで生成しています（赤のプロット）   

**Dim=1の結果**   

![dim1_gen](https://user-images.githubusercontent.com/39581094/76176452-19bdb900-61f4-11ea-9e8e-ea09980f6537.png)

**Dim=3の結果**   

![dim3_gen](https://user-images.githubusercontent.com/39581094/76176455-1cb8a980-61f4-11ea-8e9d-413c62252ca8.png)


   
