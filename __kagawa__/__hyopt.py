import json

hyperparameter = None # グローバル変数として持っておく

def initialize_hyperparameter(load_filename):
    """
    ハイパーパラメータを読み込む関数
    .jsonファイルを指定した場合、そこから値を読み込む
    指定しなかった場合、空のdictを用意するだけ
    
    hyperparameter["evaluation"]=None
    hyperparameter["evaluation_output"]=None
    hyperparameter["hyperparameter_input"]=load_filename
    hyperparameter["emission_internal_layers"]=None
    hyperparameter["transition_internal_layers"]=None
    hyperparameter["variational_internal_layers"]=None # build_nn
    hyperparameter["potential_internal_layers"]=None
    hyperparameter["potential_enabled"]=True
    hyperparameter["potential_grad_transition_enabled"]=True
    hyperparameter["potential_nn_enabled"]=True
    """
    
    global hyperparameter # グローバル変数で扱うと参照が楽そう
    hyperparameter = {} # dictの用意    
    
    if load_filename is not None: # .jsonファイルを指定した場合
        fp = open(load_filename, "r") # ファイルを開く
        hyperparameter.update(json.load(fp)) # dictをupdateする

def get_hyperparameter():
    """
    ハイパーパラメータを返す関数
    """
    
    global hyperparameter # グローバル変数を参照する
    
    if hyperparameter is None: # もしdictハイパーパラメータが設定されていない場合
#         initialize_hyperparameter(None) # 空で初期化する
        initialize_hyperparameter("./hyparam.json") # hyparam.jsonから読み込んで初期化する
        
        
    return hyperparameter # ハイパーパラメータを返す
