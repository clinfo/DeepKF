- ***make_data_all.py***

はじめにmdtrajをインストールします   
mdtraj (http://mdtraj.org/1.9.3/installation.html)   
```
conda install -c cond-forge mdtraj
```
使い方
```
python make_data_ala.py
```
***スクリプトの中身は適宜編集してください(vim等で編集します)***

スクリプト内24-26行目   

> 24     fname1_list = ['../trajectory-'+str(i)+'.dcd' for i in range(2, 9)]   
> 25     fname2_list = ['../trajectory-'+str(g)+'.xtc' for g in range(1, 6)]   
> 26     angles = get_angles_dcd("../trajectory-1.dcd")

でトラジェクトリデータ(dcdやxtc)の場所を指定します   

（トラジェクトリデータ(dcdやxtc)は、小島Google Driveの
```
traj_data/ala_data/trajectory-**.dcd
traj_data/ala_data/trajectory-**.xtc
```
から入手できます）
※計算機 pe1 の場合,
```
/data/traj_data/ala_data/trajectory-**.dcd
/data/traj_data/ala_data/trajectory-**.xtc
```
を参照する. （コピー禁止）

スクリプト内42行目で形状を指定できます

> 42     traj1 = np.reshape(angles, (-1, 300, 2))   


スクリプト内44行目で出力名を指定できます

> 44     np.save('ala_traj_all.npy', traj1)   



- ***phi_psi_sns_all.py***   
構造数のヒートマップを作成します
```
python phi_psi_sns_all.py
```

![sns_all_out](https://user-images.githubusercontent.com/39581094/75693181-f376b580-5ce9-11ea-9c8a-bd52a03e628b.png)

