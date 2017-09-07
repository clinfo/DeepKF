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
python dkf.py --config sample/config.json --hyperparam sample/hyparam.json train 
```
config.jsonファイルはデータ入出力等に関する設定を記述する設定ファイル
hyparam.jsonファイルは主に学習に関する設定を記述する設定ファイル


### 予測

```
python dkf.py --config sample/config.json --hyperparam model/hyparam.result.json --save-config ./model/config.result.json infer
```
model/hyparam.result.json
は学習時に与えられたhyparam.jsonファイルから自動決定されたパラメータ等も含めた設定ファイル


### フィルタリング

```
python dkf.py --config model/config.result.json --hyperparam model/hyparam.result.json filter
```


### 状態空間の計算

```
python attractor.py --config model/config.result.json --hyperparam model/hyparam.result.json field
```

### 予測のプロット

```
python script/plot.py model_2d/config_infer.result.json all
```

### フィルタリングのプロット
```
python script/plot_p.py ./model_2d/config_infer.result.json all 
```

### 状態空間のプロット
```
python script/plot_vec.py model/config_infer.result.json
```

