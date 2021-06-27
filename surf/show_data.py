from surf.script_tab import keytab
import os, json, time, re, codecs, glob
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import itertools
import mplfinance as mpf
import numpy as np

mpl.use('TkAgg')


# 显示区间密度
def range_density(pdseri, headstr="", subfix="png", x_show_n=30, silence=0):
    plt.figure(figsize=(12, 8))
    # sns.distplot(pdseri.values, bins=x_show_n, kde=False, color="red")
    # 核密度估计 + 统计柱状图
    sns.distplot(pdseri.dropna(), bins=x_show_n)
    # # 核密度估计
    # sns.kdeplot(pdseri.dropna())
    # # 两支股票的皮尔森相关系数
    plt.title(headstr)
    # plt.xlabel('Loyalty score', fontsize=12)
    if subfix is not None or subfix != "":
        picfile = ".".join(headstr.split(".")[:-1] + [subfix])
        plt.savefig(picfile)
    if silence == 0:
        plt.show()


# 不同特征数值 的 方差分布
def chara_diffval_std(pdobj, headstr="", subfix="png", silence=0):
    plt.figure(figsize=(8, 4))
    sns.violinplot(x="features", y="values", data=pdobj)
    plt.title(headstr)
    # plt.xlabel('Feature 3', fontsize=12)
    # plt.ylabel('Loyalty score', fontsize=12)
    plt.xticks(rotation='vertical')
    if subfix is not None or subfix != "":
        picfile = ".".join(headstr.split(".")[:-1] + [subfix])
        plt.savefig(picfile)
    if silence == 0:
        plt.show()


def plot_stock_sig(pddatas, datals, sigs, headstr=""):
    # 1. 添加额外线
    # add_plot = mpf.make_addplot(datal[['High', 'MidValue', 'Low']])
    # 2. 添加额外点
    tadd_plot = [
        mpf.make_addplot(datal) for datal in datals
    ]
    add_plot = [
        # mpf.make_addplot(sigs["sig"], scatter=True, markersize=100, marker='o', color='y'),
        mpf.make_addplot(sigs["ub"], scatter=True, markersize=100, marker='^', color='r'),
        mpf.make_addplot(sigs["us"], scatter=True, markersize=100, marker='v', color='g'),
        mpf.make_addplot(sigs["db"], scatter=True, markersize=100, marker='^', color='#ff8080'),
        mpf.make_addplot(sigs["ds"], scatter=True, markersize=100, marker='v', color='#80ff80'),
    ]
    if "volume" in pddatas.columns:
        volsig = True
    else:
        volsig = False
    mpf.plot(pddatas, type='candle', addplot=tadd_plot + add_plot, mav=(5, 10), volume=volsig, title=headstr)


def plot_curve(pddata, headstr="", subfix="png", x_show_n=30, silence=0):
    titles = pddata.columns
    x = pddata.index
    # ys=pddata
    xin = np.arange(0, len(x))
    nums = len(titles)
    colors = ["#ff0000", "#00ff00", "#0000ff", "#ffff00", "#ff00ff", "#00ffff", "#000000"] * (nums // 7 + 1)
    # 长 宽 背景颜色
    plt.figure(figsize=(12, 6), facecolor='w')
    # plt.figure(facecolor='w')
    for n, title in enumerate(titles):
        plt.plot(xin, pddata[title], color=colors[n], linestyle='-', linewidth=1.2, marker="", markersize=7,
                 markerfacecolor='b', markeredgecolor='g', label=title)
        plt.legend(loc='upper right', frameon=False)
    plt.xlabel("x", verticalalignment="top")
    plt.ylabel("y", rotation=0, horizontalalignment="right")
    # xticks = ["今天", "周五", "周六", "周日", "周一"]
    x_show_n = int(len(x) / x_show_n)
    show_inte = x_show_n if x_show_n > 0 else 1
    s_xin = [i1 for i1 in xin if i1 % show_inte == 0]
    s_x = [i1 for id1, i1 in enumerate(x) if id1 % show_inte == 0]
    plt.xticks(s_xin, s_x, rotation=90, fontsize=10)
    # plt.xticks(xin, x, rotation=90, fontsize=5)
    # yticks = np.arange(0, 500, 10)
    # plt.yticks(yticks)
    plt.title(headstr)
    # plt.grid(b=True)
    if subfix is not None or subfix != "":
        picfile = ".".join(headstr.split(".")[:-1] + [subfix])
        plt.savefig(picfile)
    if silence == 0:
        plt.show()


class DataShow(object):
    def __init__(self):
        self.funcmap = {
            "序列特征": self.show_charas,
            "预测回测": self.show_backtest,
            "回测统计": self.show_backstatic,
        }

    def show_charas(self, infile, subfix, x_show_n=30, silence=0):
        pdobj = pd.read_csv(infile, header=0, index_col="date", encoding="utf8")
        plot_curve(pdobj, headstr=infile, subfix=subfix, x_show_n=x_show_n, silence=silence)

    def show_backtest(self, infile, subfix, x_show_n=30, silence=0):
        pdobj = pd.read_csv(infile, header=0, index_col="date", encoding="utf8")
        (_, infile) = os.path.split(infile)
        pdobj.index = pd.to_datetime(pdobj.index)
        pdobj = pdobj.iloc[0:100, :]
        candle_col = ["high", "low", "close", "open", "volume"]
        # raise 33 生成多列
        sig_cols = [i1 for i1 in pdobj.columns if re.search("oper_sig$", i1)]
        strategyhead = [i1.replace("oper_sig", "") for i1 in sig_cols]
        for headname in strategyhead:
            wealthcol = "{}wealth".format(headname)
            dataasis = [pdobj[wealthcol] * pdobj["close"][0]]
            sig_col = "{}oper_sig".format(headname)
            ratio_col = "{}oper_ratio".format(headname)
            pdobj["ub"] = 0
            pdobj["us"] = 0
            pdobj["db"] = 0
            pdobj["ds"] = 0
            u_before = 0
            d_before = 0
            for i2 in range(len(pdobj)):
                if pdobj.iloc[i2][sig_col] == 1:
                    if pdobj.iloc[i2][ratio_col] > 0:
                        if pdobj[ratio_col][i2] > u_before:
                            pdobj["ub"][i2] = pdobj["close"][i2]
                            u_before = pdobj[ratio_col][i2]
                            # print("ub", u_before)
                            d_before = 0
                        elif pdobj[ratio_col][i2] < u_before:
                            pdobj["us"][i2] = pdobj["close"][i2]
                            u_before = pdobj[ratio_col][i2]
                            # print("us", u_before)
                            d_before = 0
                    else:
                        if pdobj[ratio_col][i2] > d_before:
                            pdobj["ds"][i2] = pdobj["close"][i2]
                            d_before = pdobj[ratio_col][i2]
                            # print("ds", d_before)
                            u_before = 0
                        elif pdobj[ratio_col][i2] < d_before:
                            pdobj["db"][i2] = pdobj["close"][i2]
                            d_before = pdobj[ratio_col][i2]
                            # print("db", d_before)
                            u_before = 0
            udsig = ["ub", "us", "db", "ds"]
            plot_stock_sig(pdobj[candle_col], dataasis, pdobj[udsig], headstr=infile + headname)

    def show_backstatic(self, infile, subfix, x_show_n=30, silence=0):
        filehead = ".".join(infile.split(".")[:-1])
        pdobj = pd.read_csv(infile, header=0, index_col="filename", encoding="utf8")
        strategyhead = [i1 for i1 in pdobj.columns if re.search("年化$", i1)]
        strategyhead = [i1.replace("年化", "") for i1 in strategyhead]
        # 只画4张图
        feath_sufix = ["年化", "累计", "回撤", "夏普"]
        for feathname in feath_sufix:
            tpdlist = []
            for strate in strategyhead:
                headstr = filehead + strate + feathname + "."
                range_density(pdobj[strate + feathname], headstr=headstr, subfix=subfix, x_show_n=x_show_n,
                              silence=silence)
                tpd = pdobj[[strate + feathname]]
                tpd.columns = ["values"]
                tpd["features"] = strate + feathname
                tpdlist.append(tpd)
            showpd = pd.concat(tpdlist, axis=0)
            headstr = filehead + feathname + "."
            chara_diffval_std(showpd, headstr=headstr, subfix=subfix, silence=silence)

    def __call__(self, deal_content, projectpath, subfix, silence):
        for tkey in deal_content:
            infiles = [glob.glob(os.path.join(projectpath, i2)) for i2 in deal_content[tkey]["输入数据"]]
            infiles = list(set(itertools.chain(*infiles)))  # 展开去重
            for infile in infiles:
                try:
                    x_show_n = deal_content[tkey]["x_show_n"]
                except Exception as e:
                    x_show_n = 10000
                x_show_n = 10000 if x_show_n is None else x_show_n
                self.funcmap[tkey](infile, subfix, x_show_n, silence)
        return None


show_func = {
    "图形展示": DataShow(),
}
