***DKFの入力データ（npyファイル）を作成***

- **brown_model**:ブラン運動モデルのデータセットを作成します。  
vampnetのコード(https://github.com/markovmodel/deeptime.git) を引用します。

   - make_data_adw.py：　double well potentialのブラウン運動のデータを作成します。  
     出力は、ポテンシャル図（adw_potential.png）とnpyファイル（adw_traj1.npy）です。
  
   - make_data_brown.py：　folding potentialのブラウン運動のデータを作成します。  
     出力は、ポテンシャル図（folding_energy.png）とnpyファイル（folding_2d_traj.npy）です。

- **alanine**:アラニンジペプチドモデルのデータセットを作成します。  
mdtraj(http://mdtraj.org/1.9.3/) を使用します。

   - make_data_ala.py：　dcd・xtcトラジェクトリデータを入力とし、mdtrajを用いて二面角情報を抽出したデータを作成します。  
   - phi_psi_sns_all.py：　構造数ヒートマップ(sns_all_out.png)を作成します。

- **chignolin**:シニョリンのデータセットを作成します。  
mdtrajとmsmbuilder(http://msmbuilder.org/3.8.0/) を使用します。

   - make_data_contact.py: xtcトラジェクトリ を入力とし、msmbuilderのContact featurizer(scheme='closest-heavy')を用いて、重原子間距離を抽出したデータを作成します。
   - make_data_rmsd.py: xtcトラジェクトリ を入力とし、msmbuilderのRMSD featurizerを用いて、参照構造(初期構造)に対するRMSDを抽出したデータを作成します。
   - make_shuffledata.py: 上記で出力したnpyファイルを入力とし、時系列情報をなくしたトラジェクトリを作成します。
   
- maskdata.py: 生成filteringの際、使用するマスクデータ(npyファイル)を作成します。

