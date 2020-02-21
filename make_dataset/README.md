***DKFの入力データ（npyファイル）を作成***

- **brown_model**:ブラン運動モデルのデータセットを作成します。vampnetのコード(https://github.com/markovmodel/deeptime.git) を引用します。

   - make_data_adw.py：　double well potentialのブラウン運動のデータを作成します。
     出力は、ポテンシャル図（adw_potential.png）とnpyファイル（adw_traj1.npy）です。
  
   - make_data_brown.py：　folding potentialのブラウン運動のデータを作成します。
     出力は、ポテンシャル図（folding_energy.png）とnpyファイル（folding_2d_traj.npy）です。

- **alanine**:アラニンジペプチドモデルのデータセットを作成します。

   - make_data_ala.py：　dcd・xtcトラジェクトリデータを入力とし、mdtrajを用いて二面角情報を抽出したデータを作成します。
   - phi_psi_sns_all.py：　構造数ヒートマップ(sns_all_out.png)を作成します。

- **chignolin**:シニョリンのデータセットを作成します。

