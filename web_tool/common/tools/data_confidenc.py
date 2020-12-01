import os
import itertools
import numpy as np
import pytesseract
from PIL import Image
import csv
import re
import json
from matplotlib.font_manager import *
import matplotlib.pyplot as plt
import matplotlib.dates
import matplotlib as mpl
import matplotlib.style as style
import datetime, time
import copy
import random
import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import bisect
import xmltodict
import seaborn as sns
import pandas as pd
import codecs
import numpy as np
from pylab import *

import scipy.stats as stats  # 该模块包含了所有的统计分析函数
import statistics as bstat


def static_func():
    "给出xy分布，拟合分布的参数"
    # 均匀分布
    # 二项分布
    # 负二项分布 aka 帕斯卡分布
    # 几何分布
    # 泊松分布
    # gamma分布
    # 指数分布
    # 正态分布
    # student分布
    # 卡方分布
    # F分布

    # 数据的算术平均数（“平均数”）。
    bstat.mean()
    # 快速的，浮点算数平均数。
    bstat.fmean()
    # 数据的几何平均数
    bstat.geometric_mean()
    # 数据的调和均值
    bstat.harmonic_mean()
    # 数据的中位数（中间值）
    bstat.median()
    # 数据的低中位数
    bstat.median_low()
    # 数据的高中位数
    bstat.median_high()
    # 分组数据的中位数，即第50个百分点。
    bstat.median_grouped()
    # 离散的或标称的数据的单个众数（出现最多的值）。
    bstat.mode()
    # 离散的或标称的数据的众数列表（出现最多的值）。
    bstat.multimode()
    # 将数据以相等的概率分为多个间隔。
    bstat.quantiles()

    # 根据是否是全量样本调用 p开头的参数或不带p的。
    # 拟合
    # 置信
    #   单类对比相同的 期望
    #   单类对比相同的 方差
    #   二类对比相同的 期望
    #   二类对比相同的 方差
    # 数据的总体标准差
    bstat.pstdev()
    # 数据的总体方差
    bstat.pvariance()
    # 数据的样本标准差
    bstat.stdev()
    # 数据的样本方差
    bstat.variance()


def plot_confidence(expect=0, std=1, datanum=30):
    "给出xy 和预期值，得出置信度 type=[-1,0,1]"
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示解决方案
    plt.rcParams['axes.unicode_minus'] = False  # 负号显示解决方案
    x = np.linspace(expect - 4 * std, expect + 4 * std, datanum)
    y = stats.norm.pdf(x, expect, std)
    plt.plot(x, y)
    plt.vlines(0, 0, 0.2, linestyles='--')
    plt.text(1.1, 0.18, '0')
    # plt.text(-2, 0.01, '下跌')
    # plt.text(2.5, 0.025, '上涨')
    plt.show()


def show_all_confids(xdata, alpha=0.05):
    collist = xdata.columns
    outjson = []
    funcnames = ["uniform", "norm", "student", "laplace", "rayleigh", "F", "beta", "chi2", "expon"]
    funceven = ["bernoulli", "binom", "poisson", "multinomial", "dirichlet", "geom"]
    allfuncs = funcnames + funceven
    for colname in collist:
        if colname.split("_")[0] not in allfuncs:
            continue
        statis_json = show1confids(xdata, colname=colname, alpha=alpha)
        outjson.append(statis_json)
    return outjson


def show1confids(xdata, colname="colname", alpha=0.05):
    """ 
    统计的数列，检验均值，检验方差，临界概率，概率类型，规定值
    根据alpha和类型，输出预测区间
    """
    funcnames = ["uniform", "norm", "student", "laplace", "rayleigh", "F", "beta", "chi2", "expon"]
    funceven = ["bernoulli", "binom", "poisson", "multinomial", "dirichlet", "geom"]
    allfuncs = funcnames + funceven
    funcname = None
    for ifuncname in allfuncs:
        if re.match(r'^{}_'.format(ifuncname), colname):
            funcname = ifuncname
            break
    if funcname is None:
        raise Exception("输入名不合法{}".format(colname))
    confid = 1 - alpha
    xdata = np.array(xdata)
    mean = xdata.mean()
    sstd = xdata.std()
    v = xdata.size
    statis_json = {}
    statis_json["colname"] = colname
    statis_json["mean"] = mean
    statis_json["sstd"] = sstd
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
    cent_prob = ""
    interval = ""
    lowbound = ""
    highbound = ""
    if funcname == "uniform":
        # [loc, loc + scale]
        cent_prob = stats.uniform.pdf(mean, mean - sstd * 1.733, mean + sstd * 1.733)
        lowbound = stats.norm.ppf(1 - confid, mean - sstd * 1.733, mean + sstd * 1.733)
        highbound = stats.norm.ppf(confid, mean - sstd * 1.733, mean + sstd * 1.733)
        interval = stats.uniform.interval(confid, mean - sstd * 1.733, mean + sstd * 1.733)
    elif funcname == "norm":
        cent_prob = stats.norm.pdf(mean, mean, sstd)  # 概率密度: 在0处概率密度值
        lowbound = stats.norm.ppf(1 - confid, mean, sstd)
        highbound = stats.norm.ppf(confid, mean, sstd)
        interval = stats.norm.interval(confid, mean, sstd)
    elif funcname == "laplace":
        cent_prob = stats.laplace.pdf(mean, mean, sstd)  # 概率密度: 在0处概率密度值
        lowbound = stats.laplace.ppf(1 - confid, mean, sstd)
        highbound = stats.laplace.ppf(confid, mean, sstd)
        interval = stats.laplace.interval(confid, mean, sstd)
    elif funcname == "student":
        cent_prob = stats.t.pdf(mean, v, mean, sstd)  # 概率密度: 在0处概率密度值
        lowbound = stats.t.ppf(1 - confid, v,mean, sstd)
        highbound = stats.t.ppf(confid, v,mean, sstd)
        interval = stats.t.interval(confid, v, mean, sstd)
    elif funcname == "rayleigh":
        cent_prob = stats.rayleigh.pdf(mean, mean, sstd)  # 概率密度: 在0处概率密度值
        lowbound = stats.rayleigh.ppf(1 - confid, mean, sstd)
        highbound = stats.rayleigh.ppf(confid, mean, sstd)
        interval = stats.rayleigh.interval(confid, mean, sstd)
    elif funcname == "chi2":
        # 2*mean == sstd
        cent_prob = stats.chi2.pdf(mean, v, mean, sstd)  # 概率密度: 在0处概率密度值
        lowbound = stats.chi2.ppf(1 - confid, v, mean, sstd)
        highbound = stats.chi2.ppf(confid, v, mean, sstd)
        interval = stats.chi2.interval(confid, v, mean, sstd)
    elif funcname == "expon":
        cent_prob = stats.expon.pdf(mean, mean, sstd)  # 概率密度: 在0处概率密度值
        lowbound = stats.expon.ppf(1 - confid, mean, sstd)
        highbound = stats.expon.ppf(confid, mean, sstd)
        interval = stats.expon.interval(confid, mean, sstd)
    elif funcname == "F":
        df1 = 3
        df2 = 5
        cent_prob = stats.f.pdf(mean, df1, df2, mean, sstd)  # 概率密度: 在0处概率密度值
        lowbound = stats.f.ppf(1 - confid, mean, sstd)
        highbound = stats.f.ppf(confid, mean, sstd)
        interval = stats.f.interval(confid, mean, sstd)
    elif funcname == "bernoulli":
        p = 0.3
        cent_prob = stats.bernoulli.pmf(mean, p, mean)  # 概率密度: 在0处概率密度值
    elif funcname == "binom":
        n = 5
        p = 0.3
        cent_prob = stats.binom.pmf(mean, n, p, mean)  # 概率密度: 在0处概率密度值
    elif funcname == "beta":
        a, b = 2.31, 0.627
        cent_prob = stats.beta.pdf(mean, a, b, mean, sstd)  # 概率密度: 在0处概率密度值
        lowbound = stats.beta.ppf(1 - confid, a, b, mean, sstd)
        highbound = stats.beta.ppf(confid, a, b, mean, sstd)
        interval = stats.beta.interval(confid, a, b, mean, sstd)
    elif funcname == "poisson":
        # mean == sstd
        p = 0.3
        cent_prob = stats.poisson.pmf(mean, p, mean)  # 概率密度: 在0处概率密度值
        lowbound = stats.poisson.ppf(1 - confid, mean, sstd)
        highbound = stats.poisson.ppf(confid, mean, sstd)
        interval = stats.poisson.interval(confid, mean, sstd)
    elif funcname == "geom":
        p = 0.3
        cent_prob = stats.geom.pmf(mean, p, mean)  # 概率密度: 在0处概率密度值
    else:
        pass
    statis_json["center_prob_density"] = cent_prob
    statis_json["prob_0"] = str(interval)
    statis_json["prob_n"] = lowbound
    statis_json["prob_p"] = highbound
    return statis_json


def plot_confid(expect=0, std=1, datanum=30):
    "给出xy 和预期值，得出置信度 type=[-1,0,1]"
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示解决方案
    plt.rcParams['axes.unicode_minus'] = False  # 负号显示解决方案
    x = np.linspace(expect - 4 * std, expect + 4 * std, datanum)
    y = stats.norm.pdf(x, expect, std)
    plt.plot(x, y)
    plt.vlines(0, 0, 0.2, linestyles='--')
    plt.text(1.1, 0.18, '0')
    # plt.text(-2, 0.01, '下跌')
    # plt.text(2.5, 0.025, '上涨')
    plt.show()


if __name__ == '__main__':
    confid, mean, sstd = 0.95, 58.090994185762625, 137.2101154542154
    norm_samples = stats.norm.rvs(loc=mean, scale=sstd, size=1000)
    print(confid, mean, sstd)
    lowbound = stats.norm.ppf(1 - confid, mean, sstd)
    highbound = stats.norm.ppf(confid, mean, sstd)
    print(lowbound, highbound)
    exit()
    xdata = np.linspace(-15, 5, 30)
    print(xdata)
    # 正态分布
    # # 随机生成1000个样本
    # norm_samples = stats.norm.rvs(loc=mean, scale=std, size=1000)
    statis_json = show1confids(xdata, type=0, alpha=0.1)
    print(statis_json)
    # plot_confidence(mean, sstd)
    # 1. 统计函数的 功能
    # # 随机生成1000个样本
    # norm_samples = stats.uniform.rvs(loc=mean, scale=std, size=1000)
    # # 概率密度
    # norm_samples = stats.uniform.pdf(loc=mean, scale=std, size=1000)
    # # 概率积分累计
    # norm_samples = stats.uniform.cdf(loc=mean, scale=std, size=1000)
    # # 残存函数（1-CDF）
    # norm_samples = stats.uniform.sf(loc=mean, scale=std, size=1000)
    # # 分位点函数（CDF的逆）
    # norm_samples = stats.uniform.ppf(loc=mean, scale=std, size=1000)
    # # 逆残存函数（sf的逆）
    # norm_samples = stats.uniform.isf(loc=mean, scale=std, size=1000)
    # # 对一组随机取样进行拟合，最大似然估计方法找出最适合取样数据的概率密度函数系数。
    # norm_samples = stats.uniform.fit(loc=mean, scale=std, size=1000)
    # 2. 统计函数的 种类
    # # 均匀分布
    # norm_samples = stats.uniform.rvs(loc=mean, scale=std, size=1000)
    # # 正态分布
    # norm_samples = stats.norm.rvs(loc=mean, scale=std, size=1000)
    # # student分布
    # norm_samples = stats.t.rvs(loc=mean, scale=std, size=1000)
    # # 卡方分布
    # norm_samples = stats.chi2.rvs(loc=mean, scale=std, size=1000)
    # # 指数分布
    # norm_samples = stats.expon.rvs(loc=mean, scale=std, size=1000)
    # # 对数正态分布
    # norm_samples = stats.lognorm.rvs(loc=mean, scale=std, size=1000)
    # # 二项分布
    # norm_samples = stats.binom.rvs(loc=mean, scale=std, size=1000)
    # # F分布
    # norm_samples = stats.f.rvs(loc=mean, scale=std, size=1000)
    # # beta分布
    # norm_samples = stats.beta.rvs(loc=mean, scale=std, size=1000)
    # # 泊松分布
    # norm_samples = stats.poisson.rvs(loc=mean, scale=std, size=1000)
    # # 伽马分布
    # norm_samples = stats.gamma.rvs(loc=mean, scale=std, size=1000)
    # # 求gamma置信区间 gamma(a, b)
    # CI_gamma = stats.gamma.interval(0.95, a, scale=1 / b)
    # # 超几何分布
    # norm_samples = stats.hypergeom.rvs(loc=mean, scale=std, size=1000)
    # # 柯西分布
    # norm_samples = stats.cauchy.rvs(loc=mean, scale=std, size=1000)
    # # 拉普拉斯分布
    # norm_samples = stats.laplace.rvs(loc=mean, scale=std, size=1000)
    # # 瑞利分布
    # norm_samples = stats.rayleigh.rvs(loc=mean, scale=std, size=1000)
    print("end")
