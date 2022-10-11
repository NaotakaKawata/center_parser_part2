# center_english_part2_solver

[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

センター試験英語筆記筆記試験大問２のソルバー（2017～2015年度本試験対応）

---

## Features
ロボットは東大に入れるかプロジェクトで開発した，[センター試験XMLデータ](https://21robot.org/dataset.html)のうち，英語筆記問題の大問２を解くことができる．
対応年度は2017～2015年度である．

---

## Configuration
環境
|  ライブラリ  |  バージョン  |
| ---- | ---- |
|  Anaconda  |    |
|  Python  |  3.6  |
|  CUDA  |  10.2  |
|  Pytorch  |  1.1.0  |

---

# Install

```
pip install -e .
pip install torch mxnet-cu102mkl  # Replace w/ your CUDA version; mxnet-mkl if CPU only.
```

以下の手順で必要なデータセットとライブラリを準備してください．

```
# センター試験XMLデータをダウンロード
$ wget https://21robot.org/data/center-devtest.tar.gz -P /tmp

$ tar -zxvf /tmp/center-devtest.tar.gz 

$ mkdir data/center-2017-2015/dev && cp center-devtest/Eigo/Center-2015--Main-Eigo_hikki.xml 

$ rm /tmp/center-devtest.tar.gz

# 必要なライブラリをインストール
$ conda env create -f environment.yml
```

## Quick Start
2017年度センター試験英語筆記本試験大問2の評価

```
$ python part2solver.py
```

---

## Build
ビルド方法

---

## Usage
詳しい使い方
---

## License
GitHub Changelog Generator is released under the [MIT License](http://www.opensource.org/licenses/MIT).

---

## Acknowledgements
This Docker image is based on [Rosyuku/ubuntu-rdp](https://github.com/Rosyuku/ubuntu-rdp)

---

## Feedback 
Any questions or suggestions?

You are welcome to discuss it on:

[![Tweet](https://img.shields.io/twitter/url/http/shields.io.svg?style=social)](https://twitter.com/dancing_nanachi)
---