# -*- coding: utf-8 -*-
# 加载线性回归需要的模块和库
import random, os
import numpy as np
import pandas as pd
import statsmodels.api as sm  # 最小二乘
from statsmodels.formula.api import ols  # 加载ols模型
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from pprint import pprint
import random, os
import numpy as np
import pandas as pd
import statsmodels.api as sm  # 最小二乘
from statsmodels.formula.api import ols  # 加载ols模型
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from pprint import pprint
import json
import datetime
import itertools
import re
import time
import json
import datetime
import itertools
import re, math
# from tensorflow.keras.layers import Dense
# from tensorflow.keras import Input, Model
import statsmodels.tsa.stattools as tsat
from keras.layers import Dense
from keras import Input, Model
import keras
from tcn import TCN
# , tcn_full_summary
import tcn
from statsmodels.tsa import stattools
from arch.unitroot import ADF
from statsmodels.graphics.tsaplots import *
import matplotlib.pyplot as plt
from statsmodels.tsa import arima_model
from scipy.stats import pearsonr, spearmanr
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint

debugsig = 0


def PolyLin(degree):
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree)),
            ("Linearmodel", LinearRegression())
        ])


def regression_check(x, y, test_size=0.2):
    "验证那种拟合合适, 返回 1 模型, 2 最佳分值系数, 3 所有分值系数"
    # 1. 拆分数据集
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    if debugsig == 1:
        print("all len:{}, train len:{}, test len:{}".format(len(x), len(y_train), len(y_test)))
    # 2. 遍历训练集 评估分数
    scorejson = {}
    maxdegree = 5
    if debugsig == 1:
        print("回归拟合：")
    for degr in range(1, maxdegree):
        if debugsig == 1:
            print("degree", degr)
        lin_n = PolyLin(degree=degr)
        lin_n.fit(X_train, y_train)
        # 分值 系数 截距
        scorejson[degr] = [lin_n.score(X_test, y_test), lin_n.steps[-1][-1].coef_.tolist(),
                           lin_n.steps[-1][-1].intercept_.tolist()]
    # 3. 取最高的degree
    if debugsig == 1:
        pprint(scorejson)
    best_item = sorted(scorejson.items(), key=lambda x: x[1][0])[-1]
    if debugsig == 1:
        print("best degree:{}，score:{}".format(*best_item))
    model_n = PolyLin(degree=best_item[0])
    model_n.fit(x, y)
    best_score = best_item[1][0]
    best_degree_coff = [best_item[0], best_item[1][1:]]
    return model_n, best_score, best_degree_coff, scorejson


def iter_regression4allxy(t_pandas, max_combnum=2, test_size=0.2):
    " 遍历 pandas 中的x y"
    tcol = t_pandas.columns
    xcol = [col for col in tcol if re.search("^x_", col, re.M)]
    ycol = [col for col in tcol if re.search("^y_", col, re.M)]
    fit_json = {}
    fit_json["namepair"] = []
    fit_json["best_degree_score"] = []
    fit_json["max_combnum_vali_num"] = []
    fit_json["best_degree_coff"] = []
    fit_json["all_degree_score"] = []
    for i1x in range(1, max_combnum + 1):
        for i2x in itertools.combinations(xcol, i1x):
            tname = ",".join(i2x)
            for i3y in ycol:
                tx = t_pandas[list(i2x)].values
                ty = t_pandas[[i3y]].values
                model, best_score, best_degree_coff, allscore = regression_check(tx, ty, test_size=test_size)
                fit_json["namepair"].append(tname + ",_," + i3y)
                fit_json["best_degree_score"].append(best_score)
                fit_json["max_combnum_vali_num"].append([max_combnum, test_size])
                fit_json["best_degree_coff"].append(json.dumps(best_degree_coff, ensure_ascii=False))
                fit_json["all_degree_score"].append(json.dumps(allscore, ensure_ascii=False))
                # y_predict_n = model.predict(X)
                if debugsig == 1:
                    print(datetime.datetime.today())
    return fit_json


def tcn_train(traindata, batch_size, timesteps, input_dim, modelname="default.h5"):
    """
    tcn（nb_filters=64，kernel_size=2，nb_stacks=1，
    expansions=[1，2，4，8，16，32]，
    padding='causive'，use_skip_connections=true，
    dropout_rate=0.0，return_sequences=true，
    activation='linear'，name='tcn'，
    kernel_initializer='he_normal'，use_batch_norm=false）
    """
    # https://www.cnpython.com/pypi/keras-tcn
    i = Input(batch_shape=(batch_size, timesteps, input_dim))
    o = TCN(return_sequences=False)(i)  # The TCN layers are here.
    # o = TCN(return_sequences=True)(i)
    # o = TCN(return_sequences=False)(o)
    o = Dense(1)(o)

    m = Model(inputs=[i], outputs=[o])
    m.compile(optimizer='adam', loss='mse')

    x, y = traindata
    m.fit(x, y, epochs=10, validation_split=0.2)
    if modelname:
        m.save(modelname)
    return m


def tcn_predict(predictdata, batch_size, model=None, modelname="default.h5"):
    if modelname:
        model = keras.models.load_model(modelname)  # 加载已训练好的.h5格式的keras模型
    else:
        model = model
    res = model.predict(predictdata, batch_size=10)
    return res


def time_squence(data):
    """
    arch.arch_model(y, x=None, mean='Constant', lags=0, vol='Garch', p=1, o=0, q=1, power=2.0, dist='Normal', hold_back=None)[source]
    y ({ndarray, Series, None}) – 因变量
    x ({np.array, DataFrame}, optional) –外生变量.如果没有外生变量则模型自动省略。
    mean (str, optional) – 均值模型的名称.目前支持: ‘Constant’, ‘Zero’, ‘ARX’ 以及 ‘HARX’
    lags (int or list (int), optional) –一个整数标量，用来表明滞后阶，或者使用表明滞后位置的整数列表。
    vol (str, optional) – 波动率模型的名称，目前支持: ‘GARCH’ （默认）, ‘ARCH’, ‘EGARCH’, ‘FIARCH’ 以及 ‘HARCH’。
    p (int, optional) – 对称随机数的滞后阶（译者注：即扣除均值后的部分）。
    o (int, optional) – 非对称数据的滞后阶
    q (int, optional) – 波动率或对应变量的滞后阶
    power (float, optional) – 使用GARCH或相关模型的精度
    dist (int, optional) –
    误差分布的名称，目前支持下列分布：
    
    正态分布: ‘normal’, ‘gaussian’ (default)
    学生T分布: ‘t’, ‘studentst’
    偏态学生T分布: ‘skewstudent’, ‘skewt’
    通用误差分布: ‘ged’, ‘generalized error”
    
    statsmodels.tsa.stattools.adfuller(x, maxlag=None, regression='c', autolag='AIC', store=False, regresults=False)[source]¶
     x: 序列，一维数组
     maxlag：差分次数
     regresion:{c:只有常量，
                ct:有常量项和趋势项，
                ctt:有常量项、线性和二次趋势项，
                nc:无任何选项}
     autolag:{aic or bic: default, then the number of lags is chosen to minimize the corresponding information criterium,
              None:use the maxlag,
              t-stat:based choice of maxlag. Starts with maxlag and drops a lag until the t-statistic on the last lag length is significant at the 95 % level.}
    :return: 
    """

    LjungBox = stattools.q_stat(stattools.acf(data ** 2)[1:13], len(data))
    adfretB = ADF(data)
    print(adfretB.summary().as_text())
    # print(df.head())
    aa = tsat.adfuller(data, 1)
    print(aa)
    # 将画面一分为二
    axe1 = plt.subplot(121)
    axe2 = plt.subplot(122)
    # 在第一个画面中画出序列的自相关系数图
    plot1 = plot_acf(data, lags=30, ax=axe1)
    # 在第二个画面中画出序列的偏自相关系数图
    plot2 = plot_pacf(data, lags=30, ax=axe2)

    # order表示建立的模型的阶数，c(1,0,1)表示建立的是ARMA(1,1)模型；
    # 中间的数字0表示使用原始的、未进行过差分（差分次数为0）的数据；
    # 此处我们无需考虑它
    model1 = arima_model.ARIMA(data, order=(1, 0, 1)).fit()
    model1.summary()
    model1.conf_int()
    # 绘制时间序列模拟的诊断图
    stdresid = model1.resid / math.sqrt(model1.sigma2)
    plt.plot(stdresid)
    model1.forecast(3)[0]

def correlation(X, Y):
    # 相关性
    pearson_corr, pearson_pvalue = pearsonr(X, Y)
    spearnman_corr, spearnman_pvalue = spearmanr(X, Y)
    print("Correlation : ", pearson_corr)
    # 平稳性
    pvalue = adfuller(some_series)[1]
    # 协整性
    _, pvalue, _ = coint(X, Y)
    print("Cointegration test p-value : ", pvalue)

def main():
    # 构造变量, 数据量小于150，预测准确率加速降低。
    xnum = 10
    ynum = 3
    X, Y = make_regression(n_samples=30 * 5, n_features=xnum, n_targets=ynum, noise=1.5, random_state=1)
    # print(X, Y)
    # # 2. 回归次方筛选
    # model, bestscore, allscore = regression_check(X, Y, test_size=0.2)
    # y_predict_n = model.predict(X)
    # # print(y_predict_n)
    # 3. 回归列遍历
    t_pandas = pd.DataFrame(np.hstack((X, Y)))
    colnames = ["x_col{}".format(i1) for i1 in range(xnum)]
    colnames += ["y_col{}".format(i1) for i1 in range(ynum)]
    t_pandas.columns = colnames
    # t_pandas.to_excel(os.path.join("..", "..", "test2.xlsx"), sheet_name='Sheet1', index=False, header=True,
    #                   encoding="utf-8")
    print(t_pandas)
    fit_json = iter_regression4allxy(t_pandas, max_combnum=2, test_size=0.2)
    print(fit_json)
    exit()


if __name__ == '__main__':
    batch_size, timesteps, input_dim = None, 20, 1


    def get_x_y(size=1000):
        pos_indices = np.random.choice(size, size=int(size // 2), replace=False)
        x_train = np.zeros(shape=(size, timesteps, 1))
        y_train = np.zeros(shape=(size, 1))
        x_train[pos_indices, 0] = 1.0
        y_train[pos_indices, 0] = 1.0
        return x_train, y_train


    modelname = "default.h5"
    traindata = get_x_y()
    time_squence(traindata[1])
    exit()
    model = tcn_train(traindata, batch_size, timesteps, input_dim, modelname=None)
    predictdata, _ = get_x_y()
    resdata = tcn_predict(predictdata, batch_size, model=model, modelname=None)
    print(resdata, resdata.shape)
    exit()
    main()
