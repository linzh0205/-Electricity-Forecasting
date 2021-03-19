# -Electricity-Forecasting

NCKU DSAI HW1 - Electricity Forecasting

選擇2019、2020年的1至12月的dataset訓練SVR模型，預測2021/03/22~03/29的備轉容量
1.先使用heatmap尋找與operating reserve關聯度較高的特徵

![heatmap](https://user-images.githubusercontent.com/63357025/111800306-c7cf6900-8906-11eb-94db-4adac1107136.jpeg)

2.將這些關聯度高的特徵中，刪除其資料偏差較大的數值
3.接著就可以開始訓練模型

這此使用的是scikit-learn中的SVR model，此model設有5種kernel包含linear, poly, rbf, sigmoid, precomputed
本次使用的kernel為poly，其中需設定gamma為Kernel coefficient。

