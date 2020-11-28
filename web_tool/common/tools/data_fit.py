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
import time
import json
import datetime
import itertools

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
    best_coff = best_item[1][1:]
    return model_n, best_score, best_coff, scorejson


def iter_regression4allxy(t_pandas, max_combnum=2, test_size=0.2):
    " 遍历 pandas 中的x y"
    tcol = t_pandas.columns
    xcol = [col for col in tcol if col.startswith("x_")]
    ycol = [col for col in tcol if col.startswith("y_")]
    fit_json = {}
    fit_json["namepair"] = []
    fit_json["best_degree_score"] = []
    fit_json["max_combnum_vali_num"] = []
    fit_json["all_degree_score"] = []
    fit_json["best_coff"] = []
    for i1x in range(1, max_combnum + 1):
        for i2x in itertools.combinations(xcol, i1x):
            tname = ",".join(i2x)
            for i3y in ycol:
                tx = t_pandas[list(i2x)].values
                ty = t_pandas[[i3y]].values
                model, best_score, best_coff, allscore = regression_check(tx, ty, test_size=test_size)
                fit_json["namepair"].append(tname + ",_," + i3y)
                fit_json["best_degree_score"].append(best_score)
                fit_json["max_combnum_vali_num"].append([max_combnum, test_size])
                fit_json["all_degree_score"].append(json.dumps(allscore, ensure_ascii=False))
                fit_json["best_coff"].append(best_coff)
                # y_predict_n = model.predict(X)
                if debugsig == 1:
                    print(datetime.datetime.today())
    return fit_json


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
    main()
