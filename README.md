# Deep kalman filter

Deep kalman filter の Tensor flow による実装です

- Krishnan, Rahul G., Uri Shalit, and David Sontag. "Deep kalman filters." arXiv preprint arXiv:1511.05121 (2015).
- Krishnan, Rahul G., Uri Shalit, and David Sontag. "Structured Inference Networks for Nonlinear State Space Models", In AAAI 2017

## サンプルの動かし方

以下のコマンドで動かすことが可能です．

```
$sh sample.sh
```
## サンプルデータについて
利用しているデータは以下
```
sample/sample.csv
```

この csv ファイルから以下のスクリプトを用いて学習用のデータを作成する
```
python build_sample.py
```

これにより data/ ディレクトリ以下に展開
```
data/sample_*
```



## サンプルスクリプトについて

### 学習

```
python dmm.py --config sample/config.json --hyperparam sample/hyparam.json train 
```
config.jsonファイルはデータ入出力等に関する設定を記述する設定ファイル
hyparam.jsonファイルは主に学習に関する設定を記述する設定ファイル


### 予測

```
python dmm.py --config sample/config.json --hyperparam model/hyparam.result.json --save-config ./model/config.result.json test
```
model/hyparam.result.json
は学習時に与えられたhyparam.jsonファイルから自動決定されたパラメータ等も含めた設定ファイル


### フィルタリング

```
python dmm.py --config model/config.result.json --hyperparam model/hyparam.result.json filter
```


### 状態空間の計算

```
python dmm.py --config model/config.result.json --hyperparam model/hyparam.result.json field
```

### 上述のサンプルスクリプトをまとめて実行

```
python dmm.py --config sample/config.json --hyperparam sample/hyparam.json train,test,filter,field
```
### 予測のプロット

```
python script/plot.py --config model/config.result.json --obs_dim 0 all
```

２段のプロットを作成する。
上段が推定された状態空間、下段が観測空間になっており、観測空間は１次元のみ表示している。
この例ではデフォルトで0次元目のみを出力している``--obs_dim``オプションで何次元目を出力するかを指定可能。
観測空間の各項目は、``x``が観測値、``recons``が再構築された観測値、``pred``が予測値を表している。

### フィルタリングのプロット

```
python script/plot_p.py --config model/config.result.json --num_dim 2 --obs_dim 0 all
```
３段のプロットを作成する。
上段がサンプリングされた状態空間、中段が観測空間になっており、下段は観測空間の予測値と実際のずれをプロット

状態空間は最初の２次元のみをプロットしている。
``--num_dim``オプションで何次元を出力するかを指定可能。
各次元はdim0, dim1, dim2, ...のように表示される。

また、中段は観測空間は１次元のみ表示している。
この例ではデフォルトで0次元目のみを出力している``--obs_dim``オプションで何次元目を出力するかを指定可能。
``x`` は観測値、``pred``は予測値（パーティクル）、``pred(mean)``は予測値（パーティクル）の平均、のように表示される。

下段は観測空間と同じで観測値とのずれを表示している。


### 状態空間のプロット

```
python script/plot_vec.py model/config.result.json all
```

状態空間の時間による遷移方向の表示

## パラメータ
sample/config.jsonやsample/hyparam.jsonのような形式で各種パラメータを設定することができます。
これらは実行時に
```
--config sample/config.result.json --hyperparam sample/hyparam.result.json
```
のようにして指定します。

config.jsonには複数回の実験で変化しないパラメータを設定し、hyparam.jsonは複数回の実験で変化するパラメータを設定します。
パラメータチューニングなどを行って精度の変化を観察する場合、基本的にはhyparam.jsonのパラメータのみを変化させて、config.jsonは使いまわすといった使い方を想定していますが、二つのファイルのどちらでどのパラメータを指定しても問題ありません。hyparam.jsonとconfig.jsonで、同じパラメータに異なる値をセットした場合、hyparam.jsonが優先されます。

### 各種設定項目

#### *"data_train_npy"/"data_test_npy"*
- 学習用データ/テスト用データのnumpyファイルを指定する
- データ数x時間x特徴量の時系列多次元配列データ

#### *"mask_train_npy"/"mask_test_npy"*
- 学習用データ/テスト用データの欠損の有無を表すnumpyファイルを指定する
- 0ならば欠損、1ならば欠損なし
- データ数x時間x特徴量の時系列多次元配列データ
- 省略した場合、すべて欠損無しのデータとみなされる

#### *"steps_npy"/"steps_test_npy"*
- 学習データ/テストデータの有効な時間ステップ数を保存したnumpyファイルを指定する
- データサンプル数のベクトル
- 省略した場合すべての時間が有効となる
- データごとに時間の長さが異なる可変長のデータを扱うためのデータ

#### *"data_info_json"*
各データサンプルに関する情報

#### *"batch_size"*
- バッチサイズ

#### *"dim"*
- 状態空間の次元
- 大きいと学習が難しく、結果の解釈も難しくなる。
- 小さいと観測データを十分表現できないため、再構築(recons.)の精度が悪くなり、学習もうまくいかなくなる。

#### *"epoch"*
- 最大繰り返し回数

#### *"patience"*
- Early stopping のパラメータ
- validation のロスがこの回数以上減少しない場合はそこで学習を停止する。

#### *"load_model"*
- 保存したモデルを読み込んで使用する（テスト時のみ）

#### *"save_model_path"*
- 学習したモデルの保存先

#### "save_result_train"/"save_result_test"
- 学習とテストの結果を保存する先

#### *"train_test_ratio"*
- 学習とバリデーションの割合を指定する。学習８割、バリデーションが残りのとき、``[0.8,0.2]``のように指定する。

#### *"alpha"*
- 時間方向の損失と再構成の損失
- 1の時、二つは同じ重みになり、0の時は再構成の損失のみになる。
- 潜在空間がうまく学習できない時は小さい値を設定するとよい

#### *"evaluation_output"*
- 設定と評価の値を出力するjsonファイル
- config.jsonと同じフォーマットで"evaluation"というキーが追加され、評価結果が保存されている

#### *"hyperparameter_input"*
- ハイパーパラメータファイルhyparam.jsonをこのパラメータで指定することも可能。
- --hyparamオプションを使用した場合そちらが優先される

#### *"simulation_path"*
- シミュレーションデータを保存する

#### *"potential_enabled"*
- ポテンシャルを有効にする

#### *"potential_grad_transition_enabled"*
- ポテンシャルに沿った勾配での状態遷移を有効にする

#### *"potential_nn_enabled"*
- ポテンシャルのニューラルネットを有効にする

#### *"save_result_train"/"save_result_test"/"save_result_filter"*
- 結果を保存するファイル

#### *"plot_path"*
- プロットを保存するパス

#### *"emission_internal_layers"*
- 状態空間から観測へのニューラルネットワークのアーキテクチャを設定

#### *"transition_internal_layers"*
- 状態空間(時刻t)から状態空間(時刻t+1)へのニューラルネットワークのアーキテクチャを設定

#### *"variational_internal_layers"*
- 観測から状態空間へのニューラルネットワークのアーキテクチャを設定

#### *"potential_internal_layers"*
- 状態空間からポテンシャルへのニューラルネットワークのアーキテクチャを設定

#### *"state_type"*
- 状態空間に関する設定
- normal/discreteのいずれかを指定

#### *"sampling_type"*
- 状態空間のサンプリングの方法を指定する。
- state_type=normalとした場合、none/normalのいずれかを指定
- state_type=discreteとした場合、none/gambel-max/gambel-softmax/naiveのいずれかを指定

#### *"dynamics_type"*
- 状態空間の時間発展に関する設定
- distribution/function のいずれかを指定

#### *"pfilter_type"*
- パーティクルフィルタに関する設定
- trained_dynamics/zero_dynamisc のいずれかを指定

#### *"emission_type"*
- 観測データに関する設定
- normal/binary のいずれかを指定


