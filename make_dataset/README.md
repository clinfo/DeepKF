**DKFの入力データ（npyファイル）を作成**

brown_model:ブラン運動モデルのデータセット
vampnetのコードを引用

  make_data_adw.py： double well potentialのブラウン運動のデータを作成します。
  出力は、ポテンシャル図（adw_potential.png）とnpyファイル（adw_traj1.npy）です。
  
  make_data_brown.py: folding potentialのブラウン運動のデータを作成します。
  出力は、ポテンシャル図（folding_energy.png）とnpyファイル（folding_2d_traj.npy）です。

alanine:アラニンジペプチドモデルのデータセット
dcdトラジェクトリ ファイルを入力とし、mdtrajをを用いて二面角情報を抽出します。

