# Overview

This code is based on HRCenterNet<a href=https://github.com/Tverous/HRCenterNet>[link]</a>

Training and Validation parts are changed to pytorch-lightning.

Also, adding last layer(Classifier) for classifying.

Best F1-score is 0.75 ( Iou threshold >= 0.8 ) Precision : 0.91, Recall : 0.67

# Data Architecture
```

1.Dataset( Image Format : jpg, Json Format : json )
|ㅡㅡ 1.원천데이터
|      |ㅡㅡ Style(예서, 전서, 초서, ...)
|      |       |ㅡㅡ Book Name(완구유집_01)
|      |       |     |ㅡㅡ ...
|      |       |ㅡㅡ Book Name(완구유집_02)
|      |       |     |ㅡㅡ ...
|ㅡㅡ 2.라벨링데이터
|      |ㅡㅡ Style(예서, 전서, 초서, ...)
|      |       |ㅡㅡ Book Name(완규유집_01)
|      |       |     |ㅡㅡ ...
|      |       |ㅡㅡ Book Name(완규유집_02)
|      |       |     |ㅡㅡ ...

```

# Data Statistics

You can check in assets folders, Dataset was splitted by train 8 : valid 1 : test 1

# How to run

## augmentation

If you think dataset is not a sufficient, you can run below code.

And delete '#'( line number 35, 36 ) and line number 37 in run.py 

```bash
python augment.py
```

## Train

```bash
python run.py
```

## Test

```bash
python run.py --weight_fn <weight_path> --test
```
