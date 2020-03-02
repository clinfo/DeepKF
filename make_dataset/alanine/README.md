***make_data_all.py***

はじめにmdtrajをインストールします   
http://mdtraj.org/1.9.3/installation.html
（例）
```
pip install mdtraj
```

24・25行目   
```
 24     fname1_list = ['../trajectory-'+str(i)+'.dcd' for i in range(2, 9)]
 25     fname2_list = ['../trajectory-'+str(g)+'.xtc' for g in range(1, 6)]
```
でトラジェクトリデータ(dcdやxtc)の場所を指定します   

トラジェクトリデータ(dcdやxtc)は、小島Google Driveの
```
traj_data/ala_data/trajectory-**.dcd
traj_data/ala_data/trajectory-**.xtc
```
にあります   

42行目で形状を指定できます
```
 42     traj1 = np.reshape(angles, (-1, 300, 2))
```

44行目で出力名を指定できます
```
 44     np.save('ala_traj_all.npy', traj1)
```


***phi_psi_sns_all.py***
構造数のヒートマップを作成します   
