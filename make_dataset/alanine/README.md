- ***make_data_all.py***

はじめにmdtrajをインストールします   
mdtraj (http://mdtraj.org/1.9.3/installation.html)   
```
conda install -c conda-forge mdtraj
```
使い方
```
python make_data_ala.py
```

トラジェクトリデータは計算機 pe1 の場合,
```
/data/traj_data/ala_data/trajectory-**.dcd
/data/traj_data/ala_data/trajectory-**.xtc
```
を参照する. （コピー禁止）

スクリプト内42行目で形状を指定できます

> 42     traj1 = np.reshape(angles, (-1, 300, 2))   


スクリプト内44行目で出力名を指定できます

> 44     np.save('ala_traj_all.npy', traj1)   



- ***plot_phi_psi.py***   
構造数のヒートマップを作成します
```
python plot_phi_psi.py
```

![sns_all_out](https://user-images.githubusercontent.com/39581094/75693181-f376b580-5ce9-11ea-9c8a-bd52a03e628b.png)

