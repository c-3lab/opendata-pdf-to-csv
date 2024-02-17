# opendata-pdf-to-csv

## pdfの一括ダウンロード

### 対象ページ

緊急避妊に係る取組について - 厚生労働省のウェブサイトに掲載を希望した緊急避妊にかかる対面診療が可能な産婦人科医療機関等の一覧
https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/0000186912_00002.html

1. dimのインストール

[https://github.com/c-3lab/dim?tab=readme-ov-file#install-the-dim](https://github.com/c-3lab/dim?tab=readme-ov-file#install-the-dim)

2. プロジェクト用ディレクトリを作成
```
$ mkdir project
$ cd project
```

3. プロジェクト初期化
```
$ dim init 
```

4. pdfの一括インストール
```
$ dim install -P https://www.mhlw.go.jp/stf/seisakunitsuite/bunya/0000186912_00002.html -e ".pdf" -n "shinryoujo"
```
※ 最後の２ファイルだけ産婦人科医療機関以外のpdfが混じってしまうので注意

`./data_files`以下にダウンロードされたpdfが保存される


## pdfからcsvへ変換

