# -*- coding: utf-8 -*-
import random
import numpy as np
import pandas as pd
# 加载线性回归需要的模块和库
import statsmodels.api as sm  # 最小二乘
from statsmodels.formula.api import ols  # 加载ols模型
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_regression


def polynormal_Regression(x, y, degree=1):
    "多项式 回归"
    pf = PolynomialFeatures(degree=degree)
    x_n_fit = pf.fit_transform(x)  # degree=2 一元二次的方程就转化为二元一次的方程
    lrModel = LinearRegression()
    lrModel.fit(x_n_fit, y)
    lrModel.score(x_n_fit, y)  # 模型拟合程度
    x_2_predict = pf.fit_transform([[21], [22]])
    lrModel.predict([[21], [22]])
    return lrModel


def PolyLin(degree):
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=degree)),
            ("Linearmodel", LinearRegression())
        ])


def regression_check(x, y):
    "验证那种拟合合适"
    maxdegree = 5
    for degr in range(1, maxdegree):
        lin_n = PolyLin(degree=degr)
        lin_n.fit(x, y)
        y_predict_n = lin_n.predict(x)
        print("degree", degr)
        print(lin_n)
        print(y_predict_n)


def main():
    # 构造变量
    # 训练模型
    X, Y = make_regression(n_samples=30, n_features=2, n_targets=1, noise=1.5, random_state=1)
    regression_check(X, Y)
    exit()


if __name__ == '__main__':
    main()
