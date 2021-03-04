import numpy as np
import pandas as pd
import scipy.special as sc_special
from surf.basic_mlp import *
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from collections import deque
import time


class BarStrategy(object):
    def __init__(self, cap_init, win=10, mount_init=0.0):
        self.cap_old = cap_init
        self.mount_old = mount_init
        self.cap_new = cap_init
        self.mount_new = mount_init
        self.dq = deque(maxlen=win)

    def update_info(self, newdata, olddata,mount_old):
        self.price_new = newdata[0]
        self.price_old = olddata[0]
        return self.mount_new

    def do_strategy(self):
        return

    def up_in_move(self):
        pass


def levy_flight(n, m, beta):
    """
    This function implements Levy's flight. 近似公式
    ---------------------------------------------------
    Input parameters:
        n: Number of steps 
        m: Number of dimensions
        beta: Power law index (note: 1 < beta < 2)
    Output:
        'n' levy steps in 'm' dimension
    """
    sigma_u = (sc_special.gamma(1 + beta) * np.sin(np.pi * beta / 2) / (
        sc_special.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))) ** (1 / beta)
    sigma_v = 1
    u = np.random.normal(0, sigma_u, (n, m))
    v = np.random.normal(0, sigma_v, (n, m))
    steps = u / ((np.abs(v)) ** (1 / beta))
    return steps


def generate_curve(n, m, beta, scale=0.01, plotsig=False):
    """根据levy步长，生成曲线"""
    steps = levy_flight(n, m, beta)
    steps = np.squeeze(steps)
    ys = []
    init = 1.0
    # 变换步长，使累计永为正
    for step in steps:
        ys.append(init)
        step = step * scale
        if step > 0:
            init = init * (1 + step)
        else:
            init = init / (1 - step)
        init = 1 + (init - 1) * 0.9999
    if plotsig:
        tlen = len(ys)
        x = list(range(tlen))
        titles = "levy curve"
        # 检查画图
        plot_curve(x, [ys], titles)
    return ys


def generate_simucurve():
    # todo: 待做2
    pass


def strategy_keep50(datas, keep_cap=0.5, oper_interval=1, plotsig=False):
    """根据曲线生成策略参数: 理想情况 持仓50%"""
    capital_init = 1e4
    capital_old = capital_init
    stock_mount_old = 0.0
    price_old = 0.0
    price_new = 0.0
    wealth_old = capital_old + price_new * stock_mount_old
    wealths = []
    for id1, price_new in enumerate(datas):
        # 根据昨天的总额，按今天的价格调配
        if id1 % oper_interval != 0:
            continue
        wealth_old = capital_old + price_old * stock_mount_old
        wealth_new = capital_old + price_new * stock_mount_old
        capital_new = wealth_new * keep_cap
        stock_mount_new = (wealth_old - capital_new) / price_new
        wealth_old = wealth_new
        capital_old = capital_new
        price_old = price_new
        stock_mount_old = stock_mount_new
        wealths.append(wealth_old)
    tlen = len(datas)
    ratio_all = wealth_old / capital_init
    ratio_day = pow(ratio_all, 1.0 / tlen)
    if plotsig:
        x = list(range(tlen))
        titles = ["stock", "wealth"]
        # 检查画图
        plot_curve(x, [datas, wealths], titles)
    return ratio_day


def strategy_turtle(datas, win=10, up_sell=[0.5], down_sell=[-0.1], up_buy=[0.1, 0.2], down_buy=[-0.5], plotsig=False):
    """根据曲线生成策略参数: turtle 加减仓策略"""

    # todo: 待做1
    def yield_turle_sig(dq, up_sell=[0.5], down_sell=[-0.1], up_buy=[0.1, 0.2], down_buy=[-0.5]):
        maxdq = max(dq[:-1])
        # wealth_new = capital_old + price_new * stock_mount_old
        if dq[-1] > maxdq and up_buy_index == -1:
            up_buy_index = 0
            price_in_anchor = maxdq
            price_out_anchor = dq[-1]
        if up_buy_index != -1:
            if price_out_anchor < dq[-1]:
                price_out_anchor = dq[-1]
            float_in_ratio = dq[-1] / price_in_anchor - 1
            float_out_ratio = dq[-1] / price_out_anchor - 1
            if float_in_ratio > up_buy[up_buy_index] and float_in_ratio > up_sell[0]:
                # 获利卖
                # 上向 期下 卖出
                for idus, upsell in enumerate(up_sell):
                    if float_in_ratio > upsell and idus > up_sell_index:
                        up_sell_index = idus
                # 保持资金百分比
                cap_keep = wealth_new * (up_sell_index + 1) / up_sell_length
                if cap_keep > capital_old:
                    stock_mount_new -= (cap_keep - capital_old) / dq[-1]
                    capital_new = cap_keep
                else:
                    stock_mount_new = stock_mount_old
                    capital_new = capital_old
                yield None
            elif float_in_ratio > up_buy[up_buy_index] and float_in_ratio < up_sell[0]:
                # 加仓位
                # 上向 期上 买入
                for idub, upbuy in enumerate(up_buy):
                    if float_in_ratio > upbuy and idub > up_buy_index:
                        up_buy_index = idub
                stock_mount_keep = wealth_new * (1 + up_buy_index) / up_buy_length / dq[-1]
                if stock_mount_keep > stock_mount_old:
                    capital_new -= (stock_mount_keep - stock_mount_old) * dq[-1]
                    stock_mount_new = stock_mount_keep
                else:
                    capital_new = capital_old
                    stock_mount_new = stock_mount_old
                up_buy_sig[up_buy_index] = 1
                yield None
            elif float_out_ratio < down_sell[down_sell_index]:
                # 止损
                # 下向 期下 卖出
                for idds, downsell in enumerate(down_sell):
                    if float_in_ratio > downsell and idds > down_sell_index:
                        down_sell_index = idds
                # 保持资金百分比
                cap_keep = wealth_new * (down_sell_index + 1) / down_sell_length
                if cap_keep > capital_old:
                    stock_mount_new -= (cap_keep - capital_old) / dq[-1]
                    capital_new = cap_keep
                else:
                    stock_mount_new = stock_mount_old
                    capital_new = capital_old
                yield None
            else:
                pass
        yield None

    # 初始化当前状态
    fee_static = 0.0
    fee_ratio = 0.0
    # fee_static = 5
    # fee_ratio = 2e-4
    capital_init = 1e4
    capital_old = capital_init
    stock_mount_old = 0.0
    price_old = 0.0
    price_new = 0.0
    price_in_anchor = -1.0
    up_buy_length = len(up_buy)
    up_sell_length = len(up_sell)
    down_buy_length = len(down_buy)
    down_sell_length = len(down_sell)
    up_buy_sig = [0] * up_buy_length
    up_sell_sig = [0] * up_buy_length
    down_buy_sig = [0] * down_buy_length
    down_sell_sig = [0] * down_sell_length
    wealth_old = capital_old + price_new * stock_mount_old
    wealths = []
    # 初始化判断标记
    dq = deque(maxlen=win)
    up_sell_index, down_sell_index, up_buy_index, down_buy_index = -1, -1, -1, -1
    for id1, price_new in enumerate(datas):
        dq.append([price_new])
        if len(dq) < win or price_new is None:
            continue
        up_sell_index, down_sell_index, up_buy_index, down_buy_index \
            = yield_turle_sig(dq, up_sell, down_sell, up_buy, down_buy)
        if price_in_anchor:
            up_buy_ratio = capital_old / up_buy_length
            down_sell_ratio = capital_old / down_sell_length
            up_buy_ratio = stock_mount_old / up_buy_length
            down_sell_ratio = stock_mount_old / down_sell_length
        wealth_old = capital_old + price_old * stock_mount_old
        wealth_new = capital_old + price_new * stock_mount_old
        stock_mount_new = (wealth_old - capital_new) / price_new
        # 调仓触发条件 海龟
        if up_buy_index != -1:
            # 上向 期上 买入
            pass
        elif up_sell_index != -1:
            # 上向 期下 买出
            pass
        elif down_buy_index != -1:
            # 下向 期上 买入
            pass
        elif down_sell_index != -1:
            # 下向 期下 买出
            pass
        else:
            # 跳过
            continue

        wealth_old = wealth_new
        capital_old = capital_new
        price_old = price_new
        stock_mount_old = stock_mount_new
        wealths.append(wealth_old)
    tlen = len(datas)
    ratio_all = wealth_old / capital_init
    ratio_day = pow(ratio_all, 1.0 / tlen)
    if plotsig:
        x = list(range(tlen))
        titles = ["stock", "wealth"]
        # 检查画图
        plot_curve(x, [datas, wealths], titles)
    return ratio_day


def best_capital(datas, oper_intervals=[1], plotsig=False):
    # 不同操作周期的天化，年化。交易成本。
    best_profits = {}
    for oper_interval in oper_intervals:
        print(f"oper_interval: {oper_interval}")
        final_profit = []
        for i1 in range(1, 50):
            stt = time.time()
            # keep_cap = 0.01 * i1
            keep_cap = 0.02 * i1
            print(f"  keep cap {keep_cap}")
            # 1. 固定仓位
            ratio_day = strategy_keep50(datas, keep_cap=keep_cap, oper_interval=oper_interval, plotsig=False)
            final_profit.append([keep_cap, ratio_day])
            print(" ", (time.time() - stt) / 60)
        sort_profit = sorted(final_profit, key=lambda x: x[-1])
        best_profits[oper_interval] = sort_profit[-1][-1]
    if plotsig:
        pdobj = pd.DataFrame(best_profits.items())
        print(pdobj)
        pdobj.set_index(0, inplace=True)
        pdobj[1].plot()
        plt.show()
    sort_inter = sorted(best_profits.items(), key=lambda x: x[-1])
    return sort_inter[-1]


def best_turtle_gene(datas, win=10, up_sell=[0.5], down_sell=[-0.1], up_buy=[0.1, 0.2], down_buy=[-0.5], plotsig=False):
    # 不同操作周期的天化，年化。交易成本。
    best_profits = {}
    final_profit = []
    if 1:
        for i1 in range(1, 50):
            stt = time.time()
            # 2. 海龟
            ratio_day = strategy_turtle(datas, win=win, up_sell=up_sell, down_sell=down_sell, up_buy=up_buy,
                                        down_buy=down_buy, plotsig=False)
            final_profit.append([keep_cap, ratio_day])
            print(" ", (time.time() - stt) / 60)
        sort_profit = sorted(final_profit, key=lambda x: x[-1])
        best_profits[oper_interval] = sort_profit[-1][-1]
    if plotsig:
        pdobj = pd.DataFrame(best_profits.items())
        print(pdobj)
        pdobj.set_index(0, inplace=True)
        pdobj[1].plot()
        plt.show()
    sort_inter = sorted(best_profits.items(), key=lambda x: x[-1])
    return sort_inter[-1]


def get_kde_para(datas, plotsig=False):
    pdobj = pd.DataFrame(datas)
    df = (pdobj[0].sort_values().values)[:, np.newaxis]
    grid_param = {
        'bandwidth': [i1 * 0.1 for i1 in range(35, 38)]
    }
    kde_grid = GridSearchCV(KernelDensity(), grid_param)
    kde = kde_grid.fit(df).best_estimator_
    if plotsig:
        print(kde_grid.best_params_)
        print(kde_grid.get_params())
        means = kde_grid.cv_results_['mean_test_score']
        params = kde_grid.cv_results_['params']
        for mean, param in zip(means, params):
            print("%f  with:   %r" % (mean, param))
        # fig = plt.figure()
        # plt.subplot(221)
        plt.plot(df[:, 0], np.exp(kde.score_samples(df)), '-')
        plt.show()
    return kde


def main():
    """
    1. 不同金融曲线, 最好的现金比例 为 50
    n, m, beta = 10000000, 1, 1.8
    datas = generate_curve(n, m, beta, scale=0.01, plotsig=False)
    # oper_intervals = list(range(1, 252))
    oper_intervals = [1, 5, 21, 126, 252]
    sort_inter = best_capital(datas, oper_intervals=oper_intervals, plotsig=True)
    print(sort_inter)
    2. 遗传算法找最佳的turtle策略
    best_turtle_gene
    5. 数据密度统计 拟合
    n, m, beta = 10000, 1, 1.8
    steps = levy_flight(n, m, beta)
    steps = np.squeeze(steps)
    kdepara = get_kde_para(steps, plotsig=False)
    print(kdepara)
    # range_density(pdobj, target_col=0)
    # sort_density(pdobj, target_col=0)
    :return: 判断 板块 标的
    """
    # 1. 莱维曲线 参数提取，周期曲线复制
    np.random.seed(113)
    # n, m, beta = 10000000, 1, 1.8
    # n, m, beta = 10, 1, 1.8
    n, m, beta = 10000, 1, 1.8
    datas = generate_curve(n, m, beta, scale=0.01, plotsig=False)
    print(datas[-1])
    sort_inter = best_turtle_gene(datas, win=10, up_sell=[0.5], down_sell=[-0.1], up_buy=[0.1, 0.2], down_buy=[-0.5],
                                  plotsig=False)
    print(sort_inter)
    exit()
    # 1. 历史回测，压力模拟。
    # 2. 资金净流量模型，m2延迟，GDP，国际利率差
    # 3. 资金流入，筹码集中的 交易市场，板块，个股
    # 3. 量比，委托比，内外盘 历史记录
    # 5. m0: 流通中现金，即银行体系以外的流通现金
    # 5. m1: 狭义货币供应量，即m0 + 企事业单位活期存款。
    # 5. m2: 广义货币供应量，即m1 + 企事业单位定期存款 + 居民储蓄存款。
    pass


if __name__ == '__main__':
    main()
