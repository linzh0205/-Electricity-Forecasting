
# -Electricity-Forecasting

NCKU DSAI HW1 - Electricity Forecasting

選擇台電提供過去的2019、2020年的1至3月的發電量資訊包含淨尖峰供電能力、淨尖峰用電量、民生用電、工業用電等..
並使用Support Vector Regression迴歸模型，預測2021/03/22~03/29的備轉容量(MW)。

## Data analysis ##
使用heatmap尋找與operating reserve關聯度較高的特徵

![heatmap](https://github.com/linzh0205/-Electricity-Forecasting/blob/main/heatmap.jpeg)


## Feature selection ##
從heatmap中選擇備轉容量率(%)、麥寮第二發電廠與通霄發電廠作為訓練的特徵

## Data pre-processing ##
將這些關聯度高的特徵中，刪除2筆在訓練資料集中偏差較大的數值

![clean](https://github.com/linzh0205/-Electricity-Forecasting/blob/main/clean.jpeg)


## Model training ##
這此使用的是scikit-learn中的SVR model，此model設有5種kernel包含linear, poly, rbf, sigmoid, precomputed
本次使用的kernel為poly，需設定Kernel coefficient也就是gamma為0.1、C=1e1。



## Run the code ##
將requirements.txt下載後輸入安裝套件指令:
```
conda install --yes --file requirements.txt
```
將app.py、train.csv、test.csv、submission.csv載下後(需在同資料夾內)

輸入以下指令:
```
python app.py --training train.csv --output submission.csv
```
