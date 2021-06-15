from __dmm_functions import get_default_config # configにデフォルト値を設定する
from __dmm_mode import train # train, test, filter, field

import argparse # main()の引数を操作する
import json # jsonファイルを扱う

def main():
    print("########################################################################")
    
    ##### "dmm.py"の後の引数を受け取る設定
    parser = argparse.ArgumentParser(description="Parser processing arguments on dmm.py")
#     print("parser = ", parser)
#     parser.add_argument("mode", type=str, help="WHAT'S THIS")
#     print("parser = ", parser)
    
    ### 引数の設定    
    # どのファイルからconfigを読み込むかを指定
    # --configについてはデフォルトではなしで、指定するにしても最大1つまで
    parser.add_argument("--config", type=str, default=None, nargs="?", help="config json file")
#     print("parser = ", parser)

#     # c
#     parser.add_argument("--no-config", action="store_true", help="use default setting")


    # どのファイルからhyperparameterを読み込むかを指定
    # --hyperparamについてはデフォルトではなしで、指定するにしても最大1つまで
    parser.add_argument("--hyperparam",type=str,default=None,nargs="?",help="hyperparameter json file")
    
    # どのファイルにconfigを保存するのかを指定
    # --save-configについてはデフォルトではなしで、指定するにしても最大1つまで
    parser.add_argument("--save-config", default=None, nargs="?", help="save config json file")
    
    
    # 学習、推論などのモードを指定してもらう
    # nargsを指定しないときは1個以上を必要とする
    parser.add_argument("--mode", type=str, help="train/infer/filter/field")    
    
    args = parser.parse_args() # 引数を解析する
    
#     print("args = ",args)
    print("args.config = ",args.config)
    print("args.hyperparam = ",args.hyperparam)
    print("args.save_config = ",args.save_config)
    print("args.mode = ",args.mode)
    
    ### モードのリスト
    mode_list = args.mode.split(",")
    print("mode_list = ",mode_list)
    
    ### configの読み込み
    config = get_default_config() # dict configを読み込む
    
#     if args.config is None: # main()の引数でconfigを指定しなかった場合
#         if not args.no_config: # --no-configがfalseのとき
            
    if args.config is not None: # もしconfigファイルを引数で設定した場合
        fp = open(args.config, "r") # ファイルを開く
        config.update(json.load(fp)) # configをupdate
    else: # configファイルを引数で設定しなかった場合
        print("set some config .json file")
        exit() # 強制終了
    
    for mode in mode_list: # 各モードについて
        if mode == "train": # 学習モードのとき
            train(config)
#         elif mode == "infer" or mode == "test": # inferもしくはtestの場合、inferとtestの違いがわからない
#             if args.model is not None: # main()の引数で--modelを指定している場合
#                 config["load_model"] = args.model # 改めてモデルをロードする
#             infer(config) # 推論
#         elif mode == "filter": # フィルタリングの場合
#             if args.model is not None: # main()の引数で--modelを指定している場合
#                 config["load_model"] = args.model # 改めてモデルをロードする
#             filtering(config) # フィルタリング
#         elif mode == "field": # fieldの場合
#             field(config)



    
    
if __name__ == "__main__":
    main()