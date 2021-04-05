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
    # todo: 待做1
    def __init__(self, cap_init, mount_init=0.0, win=10,
                 up_sell=[0.5], down_sell=[-0.1],
                 up_buy=[0.1, 0.2], down_buy=[-0.5]):
        self.capital_old = cap_init
        self.capital_new = cap_init
        self.mount_new = mount_init
        self.mount_old = mount_init
        self.wealth_new = cap_init
        self.wealth_old = cap_init
        self.up_sell = up_sell
        self.down_sell = down_sell
        self.up_buy = up_buy
        self.down_buy = down_buy
        self.up_sell_length = len(self.up_sell)
        self.down_sell_length = len(self.down_sell)
        self.up_buy_length = len(self.up_buy)
        self.down_buy_length = len(self.down_buy)
        self.win = win
        self.dq = deque(maxlen=self.win)
        self.price_out_anchor = None
        self.price_in_anchor = None
        self.olddata = []
        self.newdata = []
        self.up_sell_index = -1
        self.down_sell_index = -1
        self.up_buy_index = -1
        self.down_buy_index = -1
        self.old_up_sell_index = -1
        self.old_down_sell_index = -1
        self.old_up_buy_index = -1
        self.old_down_buy_index = -1
        self.active_up_sell_index = False
        self.active_down_sell_index = False
        self.active_up_buy_index = False
        self.active_down_buy_index = False
        self.float_in_ratio = 0.0
        self.float_out_ratio = 0.0
        self.turtle_sig = False

    def _get_upbuy_index(self, upbuy_index):
        # 更新才赋旧值
        firstsig = True
        for idub, upbuy in enumerate(self.up_buy):
            if self.float_in_ratio > upbuy and idub > upbuy_index:
                if firstsig:
                    self.old_up_buy_index = self.up_buy_index
                    firstsig = False
                    self.active_up_buy_index = True
                upbuy_index = idub
        return upbuy_index

    def _get_upsell_index(self, upsell_index):
        # 上卖 优先覆盖 上买
        # 更新才赋旧值
        firstsig = True
        for idub, upbuy in enumerate(self.up_buy):
            if self.float_in_ratio > upbuy and idub > upsell_index:
                if firstsig:
                    self.old_up_sell_index = self.up_sell_index
                    firstsig = False
                    self.active_up_sell_index = True
                upsell_index = idub
        return upsell_index

    def _get_downsell_index(self, downsell_index):
        # 更新才赋旧值
        firstsig = True
        for idds, downsell in enumerate(self.down_sell):
            if self.float_out_ratio < downsell and idds > downsell_index:
                if firstsig:
                    self.old_down_sell_index = self.down_sell_index
                    firstsig = False
                    self.active_down_sell_index = True
                downsell_index = idds
        return downsell_index

    def _get_downbuy_index(self, downbuy_index):
        # 更新才赋旧值
        firstsig = True
        for iddb, downbuy in enumerate(self.down_buy):
            if self.float_out_ratio < downbuy and iddb > downbuy_index:
                if firstsig:
                    self.old_down_buy_index = self.down_buy_index
                    firstsig = False
                    self.active_down_buy_index = True
                downbuy_index = iddb
        return downbuy_index

    def _update_anchor(self):
        # 1. anchor 更新
        if self.down_sell_index != -1 and self.old_down_sell_index == -1:
            # 重置
            self.price_out_anchor = max(self.dq)
        if self.price_in_anchor is not None and self.up_buy_index != -1 and self.old_up_buy_index == -1:
            self.price_in_anchor = min(self.dq)
        # 2. ratio 获取
        self.float_in_ratio = self.dq[-1] / self.price_in_anchor - 1
        self.float_out_ratio = self.dq[-1] / self.price_out_anchor - 1
        # 3. 策略只根据索引的状态 关闭
        if sum([self.up_buy_index, self.up_sell_index, self.down_buy_index, self.down_sell_index]) == -4:
            self.turtle_sig = False
        if self.price_in_anchor is None or self.price_out_anchor is None:
            self.turtle_sig = False

    def update_info(self, newdata):
        # 1. 算老值
        if newdata[0] is None:
            return True
        self.olddata = self.newdata
        self.capital_old = self.capital_new
        self.mount_old = self.mount_new
        if len(self.dq) == 0:
            self.wealth_old = self.capital_old
        else:
            self.wealth_old = self.capital_old + self.dq[-1] * self.mount_old
        self.dq.append(newdata[0])
        if len(self.dq) < self.win:
            return True
        # 2. 算新标记
        maxdq = max(list(self.dq)[:-1])
        if self.price_in_anchor is None and self.dq[-1] > maxdq:
            self.turtle_sig = True
            self.price_in_anchor = maxdq
            self.float_in_ratio = self.dq[-1] / self.price_in_anchor - 1
            self.price_out_anchor = self.dq[-1]
            self.float_out_ratio = 0.0
        if self.turtle_sig:
            # 更新 上下锚点
            self._update_anchor()
            # 有了anchor未必达标最小阈值
            # 加仓, 上向 期上 买入
            self.up_buy_index = self._get_upbuy_index(self.up_buy_index)
            # 止赢, 上向 期下 卖出
            self.up_sell_index = self._get_upsell_index(self.up_sell_index)
            # 止损, 下向 期下 卖出
            self.down_sell_index = self._get_downsell_index(self.down_sell_index)
            # 3. 策略更新
            self.do_strategy()
        # 4. 算总额
        self.wealth_new = self.capital_old + self.dq[-1] * self.mount_old
        self.mount_new = (self.wealth_old - self.capital_new) / self.dq[-1]
        self.newdata = newdata
        # self.capital_new = cap_init
        # self.mount_new = mount_init

    def do_strategy(self):
        if self.down_sell_index > -1:
            # 止损最高优先级
            # 保持资金百分比
            cap_keep = self.wealth_new * (self.down_sell_index + 1) / self.down_sell_length
            if cap_keep > self.capital_old:
                self.mount_new -= (cap_keep - self.capital_old) / self.dq[-1]
                self.capital_new = cap_keep
            else:
                self.mount_new = self.mount_old
                self.capital_new = self.capital_old
        elif self.up_sell_index > -1:
            # 预防次等优先级, 获利卖
            # 保持资金百分比
            cap_keep = self.wealth_new * (self.up_sell_index + 1) / self.up_sell_length
            if cap_keep > self.capital_old:
                self.mount_new -= (cap_keep - self.capital_old) / self.dq[-1]
                self.capital_new = cap_keep
            else:
                self.mount_new = self.mount_old
                self.capital_new = self.capital_old
        elif self.up_buy_index > -1:
            # 交易最低优先级, 加仓位
            stock_mount_keep = self.wealth_new * (1 + self.up_buy_index) / self.up_buy_length / self.dq[-1]
            if stock_mount_keep > self.mount_old:
                self.capital_new -= (stock_mount_keep - self.mount_old) * self.dq[-1]
                self.mount_new = stock_mount_keep
            else:
                self.capital_new = self.capital_old
                self.mount_new = self.mount_old
        else:
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
    # 初始化当前状态
    fee_static = 0.0
    fee_ratio = 0.0
    # fee_static = 5
    # fee_ratio = 2e-4
    capital_init = 1e4
    price_in_anchor = -1.0
    wealths = []
    keepcap = []
    # 策略初始化
    bs = BarStrategy(capital_init, mount_init=0.0, win=win,
                     up_sell=up_sell, down_sell=down_sell,
                     up_buy=up_buy, down_buy=down_buy)
    for id1, price_new in enumerate(datas):
        pass_sig = bs.update_info([price_new])
        print(pass_sig)
        if pass_sig:
            continue
        wealths.append(bs.wealth_new)
        keepcap.append(bs.capital_new / bs.wealth_new)
    tlen = len(datas)
    ratio_all = bs.wealth_new / capital_init
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
    final_ratio = None
    stt = time.time()
    # 2. 海龟
    ratio_day = strategy_turtle(datas, win=win, up_sell=up_sell, down_sell=down_sell, up_buy=up_buy,
                                down_buy=down_buy, plotsig=False)
    final_ratio = ratio_day
    print(" ", (time.time() - stt) / 60)
    return final_ratio


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
    print(datas, datas[-1])
    final_ratio = best_turtle_gene(datas, win=10, up_sell=[0.5], down_sell=[-0.1], up_buy=[0.1, 0.2], down_buy=[-0.5],
                                   plotsig=False)
    print(final_ratio)
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
