# 概要
kaggle competitionのコンペであるM5-accuracyコンペのノートです  
参加期間：~2020/6/30  
順位：459/5558 (Top 9%)
</br>  

# フォルダについて  
m5-1：最低限の加工のみを行って推論したコード  
m5-2：最終的に提出したコード（銅メダル）  
</br>  

# 必要なライブラリ  
・python 3系  
・numpy  
・pandas  
・scikit-learn  
・lightgbm  
</br>  

# 使いかた 
１．kaggleにファイルをデータセットとしてアップロード  

２．ノートにM5のデータとアップロードしたデータセットを追加
  
#### データ加工をしたい場合  
３．作業ディレクトリ(data_preprocessing)に移動  
```py
import os  
path = "..input/データセット名/m5/m5-1/data_preprocessing"  
os.chdir(path)
```

４．ファイルの実行  
```py
!python create_data.py  
```

５．出力ファイルをデータセットに
  
#### submissionファイルの作成をしたい場合  
３．作業ディレクトリ(m5-1 or m5-2)に移動  
```py
import os
path = "..input/データセット名/m5/m5-2"
os.chdir(path)
```

４．ファイルの実行  
```py
!python submission.py --data_folder データセット名
```