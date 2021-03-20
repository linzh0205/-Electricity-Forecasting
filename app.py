import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn import metrics
import seaborn as sns  #繪圖
#選擇春季的月份，包含2019、2020年的1~3月+2021年1月的dataset訓練SVR回歸模型，預測2021/03/22~03/29的備轉容量

if __name__ =='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                        default='train.csv',
                        help='input training data file name')
    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

def heatmap(train):
    #heatmap
    corr=train.corr()
    # The number of columns to be displayed in the heat map
    k = 5
    # Calculate for the top 5 columns with the highest correlation with operating reserve
    cols = corr.nlargest(k, 'operating reserve')['operating reserve'].index
    cm = np.corrcoef(train[cols].values.T)  
    # Font size of the heatmap
    sns.set(font_scale=1.25)
    # View in a heat map
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, 
                     yticklabels=cols.values, xticklabels=cols.values)
    #觀察資料分布狀況，發現有些特徵存在較大的偏差值
    sns.set()
    cols = ['operating reserve', 'rate', 'MyL#2','Tone']
    #cols = ['operating reserve', 'rate', 'MyL#2','ele_pro','people_use']
    sns.pairplot(train[cols],size=1.2)
    #plt.show()
    #將顯示前一筆最大的數據，並將其刪除，試圖減少偏差值
    train=train.drop(index=train.sort_values(by='rate',ascending=False)['date'][:2].index)
    train=train.drop(index=train.sort_values(by='MyL#2',ascending=False)['date'][:2].index)
    #train=train.drop(index=train.sort_values(by='ele_pro',ascending=False)['date'][:2].index)
    #train=train.drop(index=train.sort_values(by='people_use',ascending=False)['date'][:2].index)
    train=train.drop(index=train.sort_values(by='Tone',ascending=False)['date'][:1].index)
    #print(train)
    return train

def forecasting():
    #import data
    train = pd.read_csv(args.training)
    train_new = heatmap(train)

    #建立training dataset
    X = train_new[['rate', 'MyL#2','Tone']]
    #X = train_new[['rate', 'MyL#2','ele_pro','people_use']]
    #print(X.shape)
    Y = train_new[['operating reserve']]
    #print(Y.shape)
    Y = Y.values.reshape(-1,1) 
    #Feature scaling
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    train_x = scaler_x.fit_transform(X)
    train_y = scaler_y.fit_transform(Y)
    #SVR
    regressor = SVR(kernel='poly', C=1e1, gamma=0.01)
    regressor.fit(train_x,train_y)
    
    test = pd.read_csv('test.csv')
    #test_y = test[['operating reserve']][43:]
    #print(test_y)
    #pred_x = test[['rate', 'MyL#2','Tone']][12:19]
    #22~29
    #mean 1/23~1/29
    pred_x = test[['rate', 'MyL#2','Tone']][22:29]
    #print(pred_x)
    pred_x = scaler_x.fit_transform(pred_x)
    pred = regressor.predict(pred_x)
    pred=scaler_y.inverse_transform(pred)
    #print("RMSE:", np.sqrt(metrics.mean_squared_error(test_y, pred)))
    #print(pred)

    name = ['operating reserv(MW)']
    pred = pd.DataFrame(pred, columns=name)
    date = [['date'],['2021/3/23'],['2021/3/24'],['2021/3/25'],
              ['2021/3/26'],['2021/3/27'],['2021/3/28'],['2021/3/29']]
    name = date.pop(0)
    date_df = pd.DataFrame(date,columns=name)
    res = pd.concat([date_df,pred],axis = 1)
    res.to_csv(args.output,index=0)
    # print(res)

forecasting()
