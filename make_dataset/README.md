### ***DKFの入力データ（npyファイル）を作成***

- **brown_model**:ブラン運動モデルのデータセットを作成     
vampnetのコード(https://github.com/markovmodel/deeptime.git) を引用

   - make_data_adw.py：　assymetry double well potentialのブラウン運動のデータを作成    
     出力は、ポテンシャル図（adw_potential.png）とnpyファイル（adw_traj1.npy）  
  
   - make_data_brown.py：　folding potentialのブラウン運動のデータを作成    
     出力は、ポテンシャル図（folding_energy.png）とnpyファイル（folding_2d_traj.npy）  

- **alanine**:アラニンジペプチドモデルのデータセットを作成  
mdtraj(http://mdtraj.org/1.9.3/) を使用  

   - make_data_ala.py：　dcd・xtcトラジェクトリデータを入力とし、mdtrajを用いて二面角情報を抽出したデータを作成   
   - phi_psi_sns_all.py：　構造数ヒートマップ(sns_all_out.png)を作成  

- **chignolin**:シニョリンのデータセットを作成    
mdtrajとmsmbuilder(http://msmbuilder.org/3.8.0/) を使用  

   - make_data_contact.py: xtcトラジェクトリ を入力とし、msmbuilderのContact featurizer(scheme='closest-heavy')を用いて、重原子間距離を抽出したデータを作成  
   - make_data_rmsd.py: xtcトラジェクトリ を入力とし、msmbuilderのRMSD featurizerを用いて、参照構造(初期構造)に対するRMSDを抽出したデータを作成  
   - make_shuffledata.py: 上記で出力したnpyファイルを入力とし、時系列情報をなくしたトラジェクトリを作成  
   
- maskdata.py: 生成を行う際に使用するマスクデータ(npyファイル)を作成  

