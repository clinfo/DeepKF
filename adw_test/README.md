```
DeepKF/sample_adw
```
で上手くいかなかった場合、こちらもご確認ください  
  
sample_md2以下に1次元ブラウン運動(adwモデル)のトラジェクトリを入力し実行したDKF結果を入れてます   
  
下記コマンドで結果を確認してみてください  

```
cd adw_test
cp -r sample_md2 sample_md2_old        #実行済みの結果をsample_md2_oldに移す
rm -r sample_md2/model                 #sample_md2下の実行結果を消す
rm -r sample_md2/config.result.json　　 #sample_md2下の実行結果を消す
mkdir -p sample_md2/model  
sh sample_md2/run_sample.sh            #新たに実行
```
