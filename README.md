# SmolVLAのLoRA追加学習とLIBERO-plus評価

Physical AI講座の課題を出発点に、ロボットの視覚・言語・行動モデルSmolVLAに、Spatialタスクの成功軌道を使ってLoRAで追加学習しました。Spatialではベースと同じ30/30でした。4スイートの追加評価では、ベースの28/33に対しLoRA版は14/33でした。

**English summary.** I adapted a SmolVLA course experiment with Claude assistance, expanded the LIBERO-plus evaluation, and compared the same base policy with and without a Spatial LoRA adapter. Both reached 30/30 in the main Spatial evaluation. Across the four suites in the secondary evaluation, success fell from 28/33 to 14/33 after adaptation.

## 担当と変更点

配布Notebookを起点に、Claudeを活用して学習データの選定数、LoRA設定、評価の試行数、成功率の集計を変更しました。SmolVLA・LeRobot・LIBERO-plusは外部のモデル、ライブラリ、評価環境です。

| 項目 | 配布版の設定 | 改良版で実行した設定 |
| --- | --- | --- |
| Spatialの学習エピソード | 10タスク × 5、計50 | 10タスク × 10、計100。選定数をタスク数から検査 |
| LoRA rank / alpha | 16 / 16 | 64 / 64 |
| 学習ステップ / バッチサイズ | 3,000 / 1 | 10,000 / 8 |
| 学習率 / 減衰後の学習率 | 3e-4 / 3e-5 | 1e-3 / 1e-4。ウォームアップ100ステップ |
| 学習時の画像増強 | コマンドに指定なし | 有効 |
| 1回の推論で実行する行動ステップ数 | 50 | 評価する両モデルで10 |
| Spatial主評価 | 10タスク × 1試行 | 10タスク × 3試行。別枠の4スイート評価と集計を追加 |

実行記録の環境はGoogle Colab、NVIDIA A100 40 GB、LeRobot v0.6.0。学習はbf16設定、学習seedは42、評価seedは2026です。学習ログに10,000ステップの完了を確認しました。ベースモデルは `lerobot/smolvla_libero_plus`、学習データは `lerobot/libero_plus` です。

## 評価結果

ベースとLoRA版を同じタスクID、試行数、評価seed、行動ステップ数で評価しました。表の分母はモデルごとの試行数です。

| 評価 | ベース成功数 | LoRA成功数 | 成功率の差 |
| --- | ---: | ---: | ---: |
| Spatial主評価（10タスク × 3） | 30/30 | 30/30 | 0.0 pp |
| 4スイート評価のSpatial（タスクID 0・4・8、各1試行） | 3/3 | 3/3 | 0.0 pp |
| Object（10タスク × 1） | 7/10 | 7/10 | 0.0 pp |
| Goal（10タスク × 1） | 8/10 | 0/10 | −80.0 pp |
| LIBERO-10（10タスク × 1） | 10/10 | 4/10 | −60.0 pp |
| **4スイート合計（主評価を除く）** | **28/33（84.8%）** | **14/33（42.4%）** | **−42.4 pp** |

4スイートを等重みで平均しても87.5%から52.5%に下がりました。Spatialはベースがすでに全試行成功しており、この条件では改善を測れませんでした。GoalとLIBERO-10の低下が集計結果に表れています。複数の設定を同時に変えた単一seedの試行なので、低下の原因や他の条件での再現性は未検証です。

## 記録の照合と再集計

実行出力付きNotebookの学習・評価出力を確認し、評価アーカイブ内の10件のJSONからタスクIDと試行ごとの成否を再集計しました。主評価と4スイート評価の各比較CSV、集計CSV、Notebookの保存出力で成功数・試行数が一致しました。同時期に保存された統合モデルの設定値も、学習率・学習ステップ・行動ステップ数と整合しました。評価JSONにはモデル重みのハッシュがないため、評価結果と重みファイルの同一性は設定・出力の照合範囲です。

公開するのは上記の変更要約、実測した成功数と試行数のCSV、および公開用に書いた再集計スクリプトです。Python 3の標準ライブラリだけで、率・差・4スイートの試行数加重集計と等重み平均を再計算できます。

```sh
python summarize.py
```

- [LeRobot](https://github.com/huggingface/lerobot)・[SmolVLAベースモデル](https://huggingface.co/lerobot/smolvla_libero_plus)
- [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus)・[学習データの配布元](https://huggingface.co/datasets/lerobot/libero_plus)
- [Physical AI講座](https://weblab.t.u-tokyo.ac.jp/lecture/course-list/physical-ai/)
