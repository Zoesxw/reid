## dataset
路径如:   
```
├── naicdata
│   ├── test
│       ├── gallery_a
│       ├── query_a
│       ├── query_a_list.txt
│   ├── train
│   ├── gallery_v_list.txt
│   ├── query_v_list.txt
│   ├── train_list.txt
│   ├── train_v_list.txt
```
## train+val
```
python main.py --bnneck --open-layers 'bottleneck' 'classifier'
```
使用train_v_list.txt, query_v_list.txt, gallery_v_list.txt, 进行训练和验证集的评估, 每隔20 epochs评估一次.

## train+test
```
python test.py --bnneck --open-layers 'bottleneck' 'classifier'
```
使用train_list.txt进行训练, 训练完成后对test文件夹测试，得到json文件
