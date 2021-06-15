import numpy as np

class dotdict(dict):
    """
    dictを扱いやすくするための新たなクラス
    d["value"]以外にもd.valueでアクセスできる
    """

    __getattr__ = dict.get # d["value"]以外にもd.valueでアクセスできる
#     __setattr__ = dict.__setitem__
#     __delattr__ = dict.__delitem__


def load_data(config,with_shuffle=True,with_train_test=True,test_flag=False,output_dict_flag=True):
    """
    返り値: 
    train_data, valid_data
    train_data, valid_dataの型はdotdict、本ファイルの上で定義した、dictを扱いやすくしたクラス
    """
    
#     time_major = config["time_major"] # デフォルトではTrue
    l = None # ラベル用データ、後にデータから読み込まれない場合はずっとNone
    
    if not test_flag: # テストではない場合、デフォルトではFalse
        x = np.load(config["data_train_npy"]) # 学習用データを読み込む、デフォルトではNone
        
        if config["mask_train_npy"] is None: # マスクを指定しない場合、デフォルトではNone
            m = np.ones_like(x) # マスクの作成
        else: # マスクを指定する場合
            m = np.load(config["mask_train_npy"]) # マスクの読み込み
            
        if config["steps_train_npy"] is None: # ステップを指定しない場合、デフォルトではNone
            s = [len(x[i]) for i in range(len(x))]
            s = np.array(s)
        else: # ステップを指定する場合
            s = np.load(config["steps_train_npy"])
        
        if config["label_train_npy"] is not None: # ラベルを指定した場合、デフォルトではNone
            l = np.load(config["label_train_npy"]) # ラベルデータを読み込む
    else: # test_flag == Trueの場合、つまりテストの場合
        x = np.load(config["data_test_npy"]) # テストデータを読み込む、デフォルトではNone
        
        if config["mask_test_npy"] is None: # マスクを指定しない場合、デフォルトではNone
            m = np.ones_like(x) # マスクの作成
        else: # マスクを指定する場合
            m = np.load(config["mask_test_npy"]) # マスクの読み込み
        
        if config["steps_test_npy"] is None: # ステップを指定しない場合、デフォルトではNone
            s = [len(x[i]) for i in range(len(x))]
            s = np.array(s)
        else: # ステップを指定する場合
            s = np.load(config["steps_test_npy"])
            
        if config["label_test_npy"] is not None: # ラベルを指定した場合、デフォルトではNone
            l = np.load(config["label_test_npy"]) # ラベルデータを読み込む
            
    ##### 以降は学習時もテスト時も実行
    if not config["time_major"]: # デフォルトではTrue
        x = x.transpose((0, 2, 1)) # 次元1と次元2を入れ替える
        m = m.transpose((0, 2, 1)) # 次元1と次元2を入れ替える
    
    ##### データを学習用・テスト用に分割
    data_num = x.shape[0] # データサイズ
    data_idx = list(range(data_num)) # データのインデックス
    
    if with_shuffle: # データをシャッフルする場合、デフォルトではTrue
        np.random.shuffle(data_idx) # データをシャッフルする
        
#     sep = [0.0, 1.0]

    if with_train_test: # 学習用データとテスト用データを分割する場合、デフォルトではTrue
#         sep = config["train_test_ratio"] # デフォルトでは[0.8, 0.2]
        tr_idx = data_idx[:int(data_num*config["train_test_ratio"][0])]
        te_idx = data_idx[int(data_num*config["train_test_ratio"][0]):]
    else: # 分割しない場合
#         sep = [0.0, 1.0] # テスト100%
        tr_idx = data_idx[:0]
        te_idx = data_idx[:]
    
#     if with_train_test: # 学習用データとテスト用データを分割する場合、デフォルトではTrue
#         sep = config["train_test_ratio"] # デフォルトでは[0.8, 0.2]
#     else: # 分割しない場合
#         sep = [0.0, 1.0] # テスト100%
    
    ##### dotdict形式でデータを保持する
    train_data = dotdict() # 学習用データを持つdotdict
    valid_data = dotdict() # 検証・テスト用データを持つdotdict
    
    tr_x = x[tr_idx] # 後に多用
    te_x = x[te_idx] # 後に多用
    
    train_data.x = tr_x # 学習用データ
    valid_data.x = te_x # テスト用データ
    
    train_data.m = m[tr_idx] # 学習用データのマスク
    valid_data.m = m[te_idx] # テスト用データのマスク
    
    train_data.s = s[tr_idx] # 学習用データのステップ
    valid_data.s = s[te_idx] # テスト用データのステップ
    
    if l is not None: # もし上でラベルが読み込まれた場合
        train_data.l = l[tr_idx] # 学習用データのラベル
        valid_data.l = l[te_idx] # テスト用データのラベル
    
    # データサイズ
    train_data.num = tr_x.shape[0]
    valid_data.num = te_x.shape[0]
    
    # ステップ？時系列の長さ？
    train_data.n_steps = tr_x.shape[1]
    valid_data.n_steps = te_x.shape[1]
    
    # 次元？
    train_data.dim = tr_x.shape[2]
    valid_data.dim = tr_x.shape[2]
    
    return train_data, valid_data