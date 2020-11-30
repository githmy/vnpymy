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


def get_confidence(xdata, expect=0, std=1, prob=0.5, type=0, alpha=0.1):
    "给出x 和预期值，得出置信度 type=[-1,0,1]"
    expect = 2.6
    std = 3.1
    confid = 1 - alpha
    xdata = np.array(xdata)
    mean = xdata.mean()
    sstd = xdata.std()
    prob = stats.norm.pdf(0, expect, std)  # 在0处概率密度值
    pre = stats.norm.cdf(0, expect, std)  # 预测小于0的概率
    interval = stats.norm.interval(confid, expect, std)  # 96%置信水平的区间
    print('随机变量在0处的概率密度是{:.3f},\n    小于0的概率是{:.3f},\n    96%的置信区间是{}'.format(prob, pre, interval))
    return mean, sstd


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


def show_confids(xdata, colname="colname", expect=None, prob=0.5, type=0, alpha=0.1):
    """ 
    统计的数列，检验均值，检验方差，临界概率，概率类型，规定值
    expect 为None, 根据alpha和类型，输出预测区间
    """
    if expect is None:
        pass
    confid = 1 - alpha
    xdata = np.array(xdata)
    mean = xdata.mean()
    sstd = xdata.std()
    v = xdata.size
    print(v)
    funcnames = ["uniform", "norm", "laplace", "students", "F", "beta", "chi2", "expon", "rayleigh"]
    funceven = ["bernoulli", "binom", "poisson", "geom"]
    confidlist = [0.5, 0.9, 0.99, 0.999]
    statis_json = []
    for funcname in funcnames:
        if funcname in funceven:
            continue
        print(funcname)
        tmp_json = {}
        tmp_json["colname"] = colname
        tmp_json["funcname"] = funcname
        tmp_json["mean"] = mean
        tmp_json["sstd"] = sstd
        interval = stats.norm.interval(confid, mean, sstd)  # 样本统计结果，96%置信水平的区间
        tmp_json["prob_interval_{}".format(confid)] = str(interval)
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
        if funcname == "uniform":
            # [loc, loc + scale]
            prob = stats.uniform.pdf(mean, mean - sstd, mean + sstd)  # 概率密度: 在0处概率密度值
        elif funcname == "norm":
            prob = stats.norm.pdf(mean, mean, sstd)  # 概率密度: 在0处概率密度值
        elif funcname == "laplace":
            prob = stats.laplace.pdf(mean, mean, sstd)  # 概率密度: 在0处概率密度值
        elif funcname == "students":
            prob = stats.t.pdf(mean, v, mean, sstd)  # 概率密度: 在0处概率密度值
        elif funcname == "chi2":
            # 2*mean == sstd
            prob = stats.chi2.pdf(mean, v, mean, sstd)  # 概率密度: 在0处概率密度值
        elif funcname == "expon":
            prob = stats.expon.pdf(mean, mean, sstd)  # 概率密度: 在0处概率密度值
        elif funcname == "rayleigh":
            prob = stats.rayleigh.pdf(mean, mean, sstd)  # 概率密度: 在0处概率密度值
        elif funcname == "F":
            df1 = 3
            df2 = 5
            prob = stats.f.pdf(mean, df1, df2, mean, sstd)  # 概率密度: 在0处概率密度值
        elif funcname == "beta":
            a, b = 2.31, 0.627
            prob = stats.beta.pdf(mean, a, b, mean, sstd)  # 概率密度: 在0处概率密度值
        elif funcname == "bernoulli":
            p = 0.3
            prob = stats.bernoulli.pmf(mean, p, mean)  # 概率密度: 在0处概率密度值
        elif funcname == "binom":
            n = 5
            p = 0.3
            prob = stats.binom.pmf(mean, n, p, mean)  # 概率密度: 在0处概率密度值
        elif funcname == "poisson":
            # mean == sstd
            p = 0.3
            prob = stats.poisson.pmf(mean, p, mean)  # 概率密度: 在0处概率密度值
        elif funcname == "geom":
            p = 0.3
            prob = stats.geom.pmf(mean, p, mean)  # 概率密度: 在0处概率密度值
        else:
            prob = None
        tmp_json["center_prob_density"] = prob
        print(prob, tmp_json)
        for oneconfi in confidlist:
            if funcname == "uniform":
                interval = stats.uniform.interval(oneconfi, mean, sstd)  # 样本统计结果，96%置信水平的区间
            elif funcname == "norm":
                interval = stats.norm.interval(oneconfi, mean, sstd)  # 样本统计结果，96%置信水平的区间
            elif funcname == "laplace":
                interval = stats.laplace.interval(oneconfi, mean, sstd)
            elif funcname == "students":
                interval = stats.t.interval(oneconfi, mean, sstd)
            elif funcname == "rayleigh":
                interval = stats.rayleigh.interval(oneconfi, mean, sstd)
            elif funcname == "F":
                interval = stats.f.interval(oneconfi, mean, sstd)
            elif funcname == "beta":
                interval = stats.beta.interval(oneconfi, mean, sstd)
            elif funcname == "poisson":
                interval = stats.poisson.interval(oneconfi, mean, sstd)
            elif funcname == "chi2":
                interval = stats.chi2.interval(oneconfi, mean, sstd)
            elif funcname == "expon":
                interval = stats.expon.interval(oneconfi, mean, sstd)
            else:
                pass
            tmp_json["prob_interval_{}".format(oneconfi)] = str(interval)
        print(interval, tmp_json)
        statis_json.append(tmp_json)
    # print('随机变量在0处的概率密度是{:.3f},\n    小于0的概率是{:.3f},\n    {}%的置信区间是{}'.format(prob, pre, confid * 100, interval))
    return statis_json


def get_confid(xdata, expect=0, std=1, prob=0.5, type=0, alpha=0.1):
    "给出x 和预期值，得出置信度 type=[-1,0,1]"
    expect = 2.6
    std = 3.1
    confid = 1 - alpha
    xdata = np.array(xdata)
    mean = xdata.mean()
    sstd = xdata.std()
    prob = stats.norm.pdf(0, expect, std)  # 在0处概率密度值
    pre = stats.norm.cdf(0, expect, std)  # 预测小于0的概率
    interval = stats.norm.interval(confid, expect, std)  # 96%置信水平的区间
    print('随机变量在0处的概率密度是{:.3f},\n    小于0的概率是{:.3f},\n    96%的置信区间是{}'.format(prob, pre, interval))
    return mean, sstd


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


def get_relation(*cols):
    "给出不同列，得出相关性函数指标"
    print(*cols)


if __name__ == '__main__':
    xdata = np.linspace(-15, 5, 30)
    print(xdata)
    # 正态分布
    # # 随机生成1000个样本
    # norm_samples = stats.norm.rvs(loc=mean, scale=std, size=1000)
    # mean, sstd = get_confidence(xdata, expect=0, std=1, prob=0.5, type=0, alpha=0.1)
    statis_json = show_confids(xdata, expect=0, prob=0.5, type=0, alpha=0.1)
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
