### ***DKFの可視化スクリプト***

コマンド例
```
plot_%%.py --config config.result.json --hyperparam hyparam.result.json --limit_all *** all
```

- plot_brown2d_heatmap.py: 2次元ブラウン運動の座標を、潜在空間で色付けをして表示する。
- plot_ala_phi_psi_heatmap.py: アラニンジペプチドを二面角で表示。潜在空間で色付けをして表示する。
- plot_chignolin.py： シニョリンをD(Asp3N-Thr8O)とD(Asp3N-Gly7O)で表示。潜在空間で色付けをして表示する。
- plot_chinolin_withpot.py: 計算された潜在空間をもとに作成した擬似ポテンシャルを描く。

 analysis/dim_table.py: アラニンジペプチド の構造遷移を解析するためのスクリプト  
 analysis/sumup.py & error_visualize.py 実測値と予測値の誤差を計算・表示する。

