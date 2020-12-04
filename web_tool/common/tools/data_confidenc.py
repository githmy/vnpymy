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


def show_all_confids(xdata, prob=0.95, posit=None):
    " 1. 不同函数类型，不同输入预处理。 2. 调用核心。"
    collist = xdata.columns
    outjson = []
    # funcnames = ["uniform", "norm", "student", "laplace", "rayleigh", "F", "beta", "chi2", "expon"]
    # funceven = ["bernoulli", "binom", "poisson", "multinomial", "dirichlet", "geom"]
    compose1_cols = ["norm", "student", "laplace", "rayleigh", "binom", "poisson", ]  # 循环每列
    compose2_cols = ["F"]  # 循环每列
    compose_all_cols = ["chi2"]  # 所有一起
    back_cols = ["uniform", "beta", "expon", "bernoulli", "multinomial", "dirichlet", "geom"]
    # 按不同的处理方式对数据分组。
    for item in compose1_cols:
        c1_list = []
        for colname in collist:
            if colname.split("_")[0] == item:
                c1_list.append(colname)
        for i1x in c1_list:
            statis_json = show1confids(xdata[[i1x]], colname=[i1x], prob=prob, posit=posit)
            outjson.append(statis_json)
    for item in compose2_cols:
        c2_list = []
        for colname in collist:
            if colname.split("_")[0] == item:
                c2_list.append(colname)
        for i2x in itertools.permutations(c2_list, 2):
            i2x = list(i2x)
            statis_json = show1confids(xdata[i2x], colname=i2x, prob=prob, posit=posit)
            outjson.append(statis_json)
    for item in compose_all_cols:
        ca_list = []
        for colname in collist:
            if colname.split("_")[0] == item:
                ca_list.append(colname)
        statis_json = show1confids(xdata[ca_list], colname=ca_list, prob=prob, posit=posit)
        outjson.append(statis_json)
    return outjson


def show1confids(xdata, colname=["colname"], prob=0.95, posit=None):
    """ 
    每次处理一种函数的一个数据集。根据prob，输出预测区间; 根据prob，输出位置的上下概率。 
    """
    compose1_cols = ["norm", "student", "laplace", "rayleigh", "binom", "poisson", ]  # 循环每列
    compose2_cols = ["F"]  # 循环每列
    compose_all_cols = ["chi2"]  # 所有一起
    back_cols = ["uniform", "beta", "expon", "bernoulli", "multinomial", "dirichlet", "geom"]
    # funcnames = ["uniform", "norm", "student", "laplace", "rayleigh", "F", "beta", "chi2", "expon"]
    # funceven = ["bernoulli", "binom", "poisson", "multinomial", "dirichlet", "geom"]
    # allfuncs = funcnames + funceven
    allfuncs = compose1_cols + compose2_cols + compose_all_cols + back_cols
    if colname is None or len(colname) == 0:
        raise Exception("输入列名不合法{}".format(colname))
    funcname = None
    for ifuncname in allfuncs:
        if ifuncname == colname[0].split("_")[0]:
            funcname = ifuncname
            break
    if funcname is None:
        raise Exception("输入列名不合法{}".format(colname))
    confid = prob
    xdata = np.array(xdata)
    statis_json = {}
    statis_json["colname"] = "+".join(colname)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2.html
    if funcname in compose1_cols:
        mean = xdata.mean()
        sstd = xdata.std()
        sample_num = xdata.size
        v = sample_num - 1
        statis_json["mean"] = mean
        statis_json["sstd"] = sstd
        pass
    if funcname in compose2_cols:
        pass
    if funcname in compose_all_cols:
        pass
    cent_prob = ""
    bound_interval = ""
    bound_above = ""
    bound_below = ""
    prob_above = ""
    prob_below = ""
    mean_b_hyp_interval = ""
    mean_b_hyp_above = ""
    mean_b_hyp_below = ""
    mean_p_hyp_above = ""
    mean_p_hyp_below = ""
    if funcname == "uniform":
        # [loc, loc + scale]
        cent_prob = stats.uniform.pdf(mean, mean - sstd * 1.733, mean + sstd * 1.733)
        bound_above = stats.norm.ppf(1 - confid, mean - sstd * 1.733, mean + sstd * 1.733)
        bound_below = stats.norm.ppf(confid, mean - sstd * 1.733, mean + sstd * 1.733)
        bound_interval = stats.uniform.interval(confid, mean - sstd * 1.733, mean + sstd * 1.733)
        if posit is not None:
            prob_above = stats.uniform.sf(confid, mean - sstd * 1.733, mean + sstd * 1.733)
            prob_below = stats.uniform.cdf(confid, mean - sstd * 1.733, mean + sstd * 1.733)
    elif funcname == "norm":
        cent_prob = stats.norm.pdf(mean, mean, sstd)  # 概率密度: 在0处概率密度值
        bound_above = stats.norm.ppf(1 - confid, mean, sstd)
        bound_below = stats.norm.ppf(confid, mean, sstd)
        bound_interval = stats.norm.interval(confid, mean, sstd)
        mean_b_hyp_above = stats.norm.ppf(1 - confid, mean, sstd / sqrt(sample_num))
        mean_b_hyp_below = stats.norm.ppf(confid, mean, sstd / sqrt(sample_num))
        mean_b_hyp_interval = stats.norm.interval(confid, mean, sstd / sqrt(sample_num))
        if posit is not None:
            prob_above = stats.norm.sf(posit, mean, sstd)
            prob_below = stats.norm.cdf(posit, mean, sstd)
            mean_p_hyp_above = stats.norm.sf(posit, mean, sstd / sqrt(sample_num))
            mean_p_hyp_below = stats.norm.cdf(posit, mean, sstd / sqrt(sample_num))
    elif funcname == "student":
        cent_prob = stats.t.pdf(mean, v, mean, sstd)  # 概率密度: 在0处概率密度值
        bound_above = stats.t.ppf(1 - confid, v, mean, sstd)
        bound_below = stats.t.ppf(confid, v, mean, sstd)
        bound_interval = stats.t.interval(confid, v, mean, sstd)
        mean_b_hyp_above = stats.norm.ppf(1 - confid, v, mean, sstd / sqrt(sample_num))
        mean_b_hyp_below = stats.norm.ppf(confid, v, mean, sstd / sqrt(sample_num))
        mean_b_hyp_interval = stats.norm.interval(confid, v, mean, sstd / sqrt(sample_num))
        if posit is not None:
            prob_above = stats.t.sf(confid, v, mean, sstd)
            prob_below = stats.t.cdf(confid, v, mean, sstd)
            mean_p_hyp_above = stats.t.sf(posit, v, mean, sstd / sqrt(sample_num))
            mean_p_hyp_below = stats.t.cdf(posit, v, mean, sstd / sqrt(sample_num))
    elif funcname == "laplace":
        cent_prob = stats.laplace.pdf(mean, mean, sstd)  # 概率密度: 在0处概率密度值
        bound_above = stats.laplace.ppf(1 - confid, mean, sstd)
        bound_below = stats.laplace.ppf(confid, mean, sstd)
        bound_interval = stats.laplace.interval(confid, mean, sstd)
        if posit is not None:
            prob_above = stats.laplace.sf(confid, mean, sstd)
            prob_below = stats.laplace.cdf(confid, mean, sstd)
    elif funcname == "rayleigh":
        cent_prob = stats.rayleigh.pdf(mean, mean, sstd)  # 概率密度: 在0处概率密度值
        bound_above = stats.rayleigh.ppf(1 - confid, mean, sstd)
        bound_below = stats.rayleigh.ppf(confid, mean, sstd)
        bound_interval = stats.rayleigh.interval(confid, mean, sstd)
        if posit is not None:
            prob_above = stats.rayleigh.sf(confid, mean, sstd)
            prob_below = stats.rayleigh.cdf(confid, mean, sstd)
    elif funcname == "chi2":
        # 2*mean == sstd
        cent_prob = stats.chi2.pdf(mean, v, mean, sstd)  # 概率密度: 在0处概率密度值
        bound_above = stats.chi2.ppf(1 - confid, v, mean, sstd)
        bound_below = stats.chi2.ppf(confid, v, mean, sstd)
        bound_interval = stats.chi2.interval(confid, v, mean, sstd)
        if posit is not None:
            prob_above = stats.chi2.sf(confid, v, mean, sstd)
            prob_below = stats.chi2.cdf(confid, v, mean, sstd)
    elif funcname == "expon":
        cent_prob = stats.expon.pdf(mean, mean, sstd)  # 概率密度: 在0处概率密度值
        bound_above = stats.expon.ppf(1 - confid, mean, sstd)
        bound_below = stats.expon.ppf(confid, mean, sstd)
        bound_interval = stats.expon.interval(confid, mean, sstd)
        if posit is not None:
            prob_above = stats.expon.sf(confid, mean, sstd)
            prob_below = stats.expon.cdf(confid, mean, sstd)
    elif funcname == "F":
        df1 = 3
        df2 = 5
        print("xdata.shape", xdata.shape)
        cent_prob = stats.f.pdf(mean, df1, df2, mean, sstd)  # 概率密度: 在0处概率密度值
        bound_above = stats.f.ppf(1 - confid, mean, sstd)
        bound_below = stats.f.ppf(confid, mean, sstd)
        bound_interval = stats.f.interval(confid, mean, sstd)
        if posit is not None:
            prob_above = stats.f.sf(confid, mean, sstd)
            prob_below = stats.f.cdf(confid, mean, sstd)
    elif funcname == "bernoulli":
        p = 0.3
        cent_prob = stats.bernoulli.pmf(mean, p, mean)  # 概率密度: 在0处概率密度值
    elif funcname == "binom":
        norm_samples = copy.deepcopy(xdata)
        norm_samples[norm_samples[:] > 0.5] = 1
        norm_samples[norm_samples[:] <= 0.5] = 0
        posi_num = sum(norm_samples)
        p = posi_num / sample_num
        cent_prob = stats.binom.pmf(posi_num, sample_num, p)  # 概率密度: 在0处概率密度值
        bound_above = stats.binom.ppf(1 - confid, sample_num, p)
        bound_below = stats.binom.ppf(confid, sample_num, p)
        bound_interval = stats.binom.interval(confid, sample_num, p)  # 概率密度: 在0处概率密度值
        if posit is not None:
            prob_above = stats.binom.sf(posit, sample_num, p)  # 概率密度: 在0处概率密度值
            prob_below = stats.binom.cdf(posit, sample_num, p)  # 概率密度: 在0处概率密度值
    elif funcname == "beta":
        a, b = 2.31, 0.627
        cent_prob = stats.beta.pdf(mean, a, b, mean, sstd)  # 概率密度: 在0处概率密度值
        bound_above = stats.beta.ppf(1 - confid, a, b, mean, sstd)
        bound_below = stats.beta.ppf(confid, a, b, mean, sstd)
        bound_interval = stats.beta.interval(confid, a, b, mean, sstd)
        if posit is not None:
            prob_above = stats.beta.sf(confid, a, b, mean, sstd)
            prob_below = stats.beta.cdf(confid, a, b, mean, sstd)
    elif funcname == "poisson":
        # mean == sstd
        alltime = sum(xdata)
        mean = sample_num / alltime
        cent_prob = stats.poisson.pmf(mean, mean)  # 概率密度: 在mean处概率密度值
        bound_above = stats.poisson.ppf(1 - confid, mean)
        bound_below = stats.poisson.ppf(confid, mean)
        bound_interval = stats.poisson.interval(confid, mean)
        if posit is not None:
            prob_above = stats.poisson.sf(confid, mean)
            prob_below = stats.poisson.cdf(confid, mean)
    elif funcname == "geom":
        p = 0.3
        cent_prob = stats.geom.pmf(mean, p, mean)  # 概率密度: 在0处概率密度值
    else:
        pass
    statis_json["center_prob_density"] = cent_prob
    statis_json["bound_interval"] = str(bound_interval)
    statis_json["bound_above"] = bound_above
    statis_json["bound_below"] = bound_below
    statis_json["prob_above"] = prob_above
    statis_json["prob_below"] = prob_below
    statis_json["mean_b_hyp_interval"] = str(mean_b_hyp_interval)
    statis_json["mean_b_hyp_above"] = mean_b_hyp_above
    statis_json["mean_b_hyp_below"] = mean_b_hyp_below
    statis_json["mean_p_hyp_above"] = mean_p_hyp_above
    statis_json["mean_p_hyp_below"] = mean_p_hyp_below
    # print(statis_json)
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
    posit = 100
    sample_num = 1000
    norm_samples = stats.norm.rvs(loc=mean, scale=sstd, size=sample_num)
    print(confid, mean, sstd, 1000)
    n = 5
    norm_samples[norm_samples[:] > 0.5] = 1
    norm_samples[norm_samples[:] <= 0.5] = 0
    # print(sum(norm_samples), norm_samples)
    posi_num = sum(norm_samples)
    p = posi_num / sample_num
    cent_prob = stats.binom.pmf(posi_num, sample_num, p)  # 概率密度: 在0处概率密度值
    bound_above = stats.binom.ppf(1 - confid, sample_num, p)
    bound_below = stats.binom.ppf(confid, sample_num, p)
    bound_interval = stats.binom.interval(confid, sample_num, p)  # 概率密度: 在0处概率密度值
    prob_above = stats.binom.sf(posi_num, sample_num, p)  # 概率密度: 在0处概率密度值
    prob_below = stats.binom.cdf(posi_num, sample_num, p)  # 概率密度: 在0处概率密度值
    print(posi_num, sample_num, p)
    print(bound_above, bound_below, bound_interval)
    print(cent_prob, prob_above, prob_below)
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
