import os
import numpy as np
import pandas as pd
import scipy.special as sc_special
from surf.basic_mlp import *
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from collections import deque
import time
import codecs

pd.set_option('display.max_columns', None)


class BarStrategy(object):
    def __init__(self, cap_init, mount_init=0.0, win=10,
                 up_sell=[0.5], down_sell=[-0.1],
                 up_buy=[0.1, 0.2], down_buy=[-0.5]):
        self.capital_init = cap_init
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
        self.current_id = 0
        self.up_sell_length = len(self.up_sell)
        self.down_sell_length = len(self.down_sell)
        self.up_buy_length = len(self.up_buy)
        self.down_buy_length = len(self.down_buy)
        self.win = win
        self.dq = deque(maxlen=self.win)
        self.olddata = []
        self.newdata = []
        self._trade_sig_reset()

    def _trade_sig_reset(self):
        self.old_touch_up_sell_index = -1
        self.old_touch_down_sell_index = -1
        self.old_touch_up_buy_index = -1
        self.old_touch_down_buy_index = -1
        self.new_touch_up_sell_index = -1
        self.new_touch_down_sell_index = -1
        self.new_touch_up_buy_index = -1
        self.new_touch_down_buy_index = -1
        self.old_trade_up_sell_index = -1
        self.old_trade_down_sell_index = -1
        self.old_trade_up_buy_index = -1
        self.old_trade_down_buy_index = -1
        self.new_trade_up_sell_index = -1
        self.new_trade_down_sell_index = -1
        self.new_trade_up_buy_index = -1
        self.new_trade_down_buy_index = -1
        # 当天是否更新了索引
        self.float_in_ratio = 0.0
        self.float_out_ratio = 0.0
        self.price_out_anchor = None
        self.price_in_anchor = None
        self.turtle_sig = False
        self.old_up_sell_ratio = 0.0
        self.old_down_sell_ratio = 0.0
        self.old_up_buy_ratio = 0.0
        self.old_down_buy_ratio = 0.0
        self.new_up_sell_ratio = 0.0
        self.new_down_sell_ratio = 0.0
        self.new_up_buy_ratio = 0.0
        self.new_down_buy_ratio = 0.0

    def get_upbuy_index(self):
        self.old_touch_up_buy_index = self.new_touch_up_buy_index
        for idub, upbuy in enumerate(self.up_buy):
            if self.float_in_ratio > upbuy:
                self.new_touch_up_buy_index = idub
        # 如果新旧不同，设为买入指标
        if self.new_touch_up_buy_index > self.old_touch_up_buy_index:
            self.new_trade_up_buy_index = self.new_touch_up_buy_index

    def get_upsell_index(self):
        self.old_touch_up_sell_index = self.new_touch_up_sell_index
        for idus, upsell in enumerate(self.up_sell):

            if self.float_in_ratio > upsell:
                self.new_touch_up_sell_index = idus
        # 如果新旧不同，设为买入指标
        if self.new_touch_up_sell_index > self.old_touch_up_sell_index:
            self.new_trade_up_sell_index = self.new_touch_up_sell_index

    def get_downsell_index(self):
        self.old_touch_down_sell_index = self.new_touch_down_sell_index
        for idds, downsell in enumerate(self.down_sell):
            if self.float_out_ratio < downsell:
                self.new_touch_down_sell_index = idds
        # 如果新旧不同，设为买入指标
        if self.new_touch_down_sell_index > self.old_touch_down_sell_index:
            self.new_trade_down_sell_index = self.new_touch_down_sell_index

    def get_downbuy_index(self):
        self.old_touch_down_buy_index = self.new_touch_down_buy_index
        for iddb, downbuy in enumerate(self.down_buy):
            if self.float_out_ratio < downbuy:
                self.new_touch_down_buy_index = iddb
        # 如果新旧不同，设为买入指标
        if self.new_touch_down_buy_index > self.old_touch_down_buy_index:
            self.new_trade_down_buy_index = self.new_touch_down_buy_index

    def _update_anchor(self):
        # 前提是 已经进入策略状态
        # 1. anchor 更新
        # if self.new_trade_down_sell_index != -1 or self.new_trade_down_buy_index != -1:
        self.price_out_anchor = max(list(self.dq) + [self.price_out_anchor])
        # if self.price_in_anchor is not None and self.new_trade_up_buy_index != -1 and self.old_touch_up_buy_index == -1:
        #     self.price_in_anchor = min(self.dq+[self.price_in_anchor])
        # if self.new_trade_up_buy_index != -1 and self.old_touch_up_buy_index == -1:
        #     self.price_in_anchor = min(self.dq+[self.price_in_anchor])
        # 2. ratio 获取
        self.float_in_ratio = self.dq[-1] / self.price_in_anchor - 1
        self.float_out_ratio = self.dq[-1] / self.price_out_anchor - 1

    def _shrink_index(self):
        # 根据规则缩并 trade 只取一个，取较大的比率
        if self.new_touch_down_sell_index > -1 or self.new_touch_up_sell_index > -1:
            self.new_trade_up_buy_index = -1
            self.new_trade_down_buy_index = -1
            self.old_up_sell_ratio = self.new_up_sell_ratio
            self.old_down_sell_ratio = self.new_down_sell_ratio
            self.new_up_sell_ratio = (self.new_trade_up_sell_index + 1) / self.up_sell_length
            self.new_down_sell_ratio = (self.new_trade_down_sell_index + 1) / self.down_sell_length
            if self.new_up_sell_ratio < self.new_down_sell_ratio:
                # 比率小于之前，不做标记
                if self.old_down_sell_ratio >= self.new_down_sell_ratio:
                    self.new_trade_down_sell_index = -1
                self.new_trade_up_sell_index = -1
            else:
                # 比率小于之前，不做标记
                if self.old_up_sell_ratio >= self.new_up_sell_ratio:
                    self.new_trade_up_sell_index = -1
                self.new_trade_down_sell_index = -1
        else:
            self.old_up_buy_ratio = self.new_up_buy_ratio
            self.old_down_buy_ratio = self.new_down_buy_ratio
            self.new_up_buy_ratio = (self.new_trade_up_buy_index + 1) / self.up_buy_length
            self.new_down_buy_ratio = (self.new_trade_down_buy_index + 1) / self.down_buy_length
            if self.new_up_buy_ratio < self.new_down_buy_ratio:
                if self.old_down_buy_ratio >= self.new_down_buy_ratio:
                    self.new_down_buy_ratio = -1
                self.new_trade_up_buy_index = -1
            else:
                if self.old_up_buy_ratio >= self.new_up_buy_ratio:
                    self.new_up_buy_ratio = -1
                self.new_trade_down_buy_index = -1
        # 新trade 不为-1 且高于上一次的索引， 旧trade 才赋值
        if self.new_trade_down_sell_index > self.old_trade_down_sell_index:
            self.old_trade_down_sell_index = self.new_trade_down_sell_index
        else:
            self.new_trade_down_sell_index = -1
        if self.new_trade_down_buy_index > self.old_trade_down_buy_index:
            self.old_trade_down_buy_index = self.new_trade_down_buy_index
        else:
            self.new_trade_down_buy_index = -1
        if self.new_trade_up_sell_index > self.old_trade_up_sell_index:
            self.old_trade_up_sell_index = self.new_trade_up_sell_index
        else:
            self.new_trade_up_sell_index = -1
        if self.new_trade_up_buy_index > self.old_trade_up_buy_index:
            self.old_trade_up_buy_index = self.new_trade_up_buy_index
        else:
            self.new_trade_up_buy_index = -1
        # 如果没买，就当已经卖过了，不能再卖, 提前重置标记。
        if self.old_trade_down_buy_index < 0 and self.old_trade_up_buy_index < 0:
            if self.new_trade_up_sell_index > -1 or self.new_trade_down_sell_index > -1:
                self._trade_sig_reset()

    def update_check_reset(self):
        if self.new_trade_down_sell_index + 1 == self.down_sell_length or self.new_trade_up_sell_index + 1 == self.up_sell_length:
            self._trade_sig_reset()

    def update_wealth(self, newdata):
        # 1. 算老值
        self.current_id += 1
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
        self.wealth_new = self.capital_old + self.dq[-1] * self.mount_old
        if len(self.dq) < self.win:
            return True
        # 2. 算新标记
        maxdq = max(list(self.dq)[:-1])
        if self.price_in_anchor is None and self.dq[-1] > maxdq:
            # 初始设置，记录策略标记
            maxdq = max(list(self.dq)[:-1])
            self.price_in_anchor = maxdq
            self.float_in_ratio = self.dq[-1] / self.price_in_anchor - 1
            self.price_out_anchor = self.dq[-1]
            self.float_out_ratio = 0.0
            self.turtle_sig = True
        if self.turtle_sig:
            # 更新 上下锚点
            self._update_anchor()
            # 有了anchor未必达标最小阈值
            # 加仓, 上向 期上 买入
            # self.new_trade_up_buy_index = self._get_upbuy_index(self.new_trade_up_buy_index)
            self.get_upbuy_index()
            # 止赢, 上向 期下 卖出
            # self.new_trade_up_sell_index = self._get_upsell_index(self.new_trade_up_sell_index)
            self.get_upsell_index()
            # 止损, 下向 期下 卖出
            # self.new_trade_down_sell_index = self._get_downsell_index(self.new_trade_down_sell_index)
            self.get_downsell_index()
            self.get_downbuy_index()
            # 3. 策略标记缩并
            self._shrink_index()
            # if self.turtle_sig:
            # if False:
            #     print("current_id", self.current_id)
            #     print(self.old_touch_up_buy_index, self.old_touch_up_sell_index, self.old_touch_down_sell_index,
            #           self.old_touch_down_buy_index)
            #     print(self.new_touch_up_buy_index, self.new_touch_up_sell_index, self.new_touch_down_sell_index,
            #           self.new_touch_down_buy_index)
            #     print(self.old_trade_up_buy_index, self.old_trade_up_sell_index, self.old_trade_down_sell_index,
            #           self.old_trade_down_buy_index)
            #     print(self.new_trade_up_buy_index, self.new_trade_up_sell_index, self.new_trade_down_sell_index,
            #           self.new_trade_down_buy_index)
            #     print(self.dq[-1], self.price_in_anchor, self.price_out_anchor, self.float_in_ratio,
            #           self.float_out_ratio)
            # 5. 策略操作更新
            self.do_strategy()
        # 4. 算总额
        self.newdata = newdata
        # print("current_id", self.current_id)
        # print(self.wealth_new / self.capital_init, self.mount_new, newdata)
        # self.capital_new = cap_init
        # self.mount_new = mount_init

    def do_strategy(self):
        # 止损最高优先级
        if self.new_trade_down_sell_index > -1:
            # 保持资金百分比
            self.capital_new = self.wealth_new * self.new_down_sell_ratio
            self.mount_new = self.wealth_new * (1 - self.new_down_sell_ratio) / self.dq[-1]
        elif self.new_trade_up_sell_index > -1:
            # 预防次等优先级, 获利卖
            self.capital_new = self.wealth_new * self.new_up_sell_ratio
            self.mount_new = self.wealth_new * (1 - self.new_up_sell_ratio) / self.dq[-1]
        elif self.new_trade_up_buy_index > -1:
            # 交易最低优先级, 加仓位
            self.capital_new = self.wealth_new * self.new_up_buy_ratio
            self.mount_new = self.wealth_new * (1 - self.new_up_buy_ratio) / self.dq[-1]
        elif self.new_trade_down_buy_index > -1:
            # 交易最低优先级, 加仓位
            self.capital_new = self.wealth_new * self.new_down_buy_ratio
            self.mount_new = self.wealth_new * (1 - self.new_down_buy_ratio) / self.dq[-1]
        else:
            self.capital_new = self.capital_old
            self.mount_new = self.mount_new


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


class LiveCurve(object):
    # todo: 待做
    def __init__(self, n):
        self.n = n
        # self.a = 0

    def gene_bar(self):
        bar_list = []
        for i in range(self.n):
            bar_list.append(1)
        return bar_list


def generate_simucurve(n, plotsig=False):
    lc = LiveCurve(n)
    bar_list = lc.gene_bar()
    if plotsig:
        x = list(range(n))
        titles = ["simu curve"]
        # 检查画图
        plot_curve(x, [bar_list], titles)


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
    wealths = []
    keepcap = []
    # 策略初始化
    bs = BarStrategy(capital_init, mount_init=0.0, win=win,
                     up_sell=up_sell, down_sell=down_sell,
                     up_buy=up_buy, down_buy=down_buy)
    tmpfile1 = os.path.join(os.path.expanduser('~'), "tmp1.txt")
    outhand1 = codecs.open(tmpfile1, "w", "utf8")
    tmpfile2 = os.path.join(os.path.expanduser('~'), "tmp2.txt")
    outhand2 = codecs.open(tmpfile2, "w", "utf8")
    tmpfile3 = os.path.join(os.path.expanduser('~'), "tmp3.txt")
    outhand3 = codecs.open(tmpfile3, "w", "utf8")
    tmpfile4 = os.path.join(os.path.expanduser('~'), "tmp4.txt")
    outhand4 = codecs.open(tmpfile4, "w", "utf8")
    tmpfile5 = os.path.join(os.path.expanduser('~'), "tmp5.txt")
    outhand5 = codecs.open(tmpfile5, "w", "utf8")
    tmpfile6 = os.path.join(os.path.expanduser('~'), "tmp6.txt")
    outhand6 = codecs.open(tmpfile6, "w", "utf8")
    tmpfile7 = os.path.join(os.path.expanduser('~'), "tmp7.txt")
    outhand7 = codecs.open(tmpfile7, "w", "utf8")
    for id1, price_new in enumerate(datas):
        # 1. 关键更新
        pass_sig = bs.update_wealth([price_new])
        if not pass_sig:
            wealths.append(bs.wealth_new)
            keepcap.append(bs.capital_new / bs.wealth_new)
        outhand1.write("{}\n".format(bs.dq[-1] if bs.turtle_sig else None))
        outhand2.write("{}\n".format(bs.dq[-1] if bs.new_trade_up_buy_index > -1 else None))
        outhand3.write("{}\n".format(bs.dq[-1] if bs.new_trade_up_sell_index > -1  else None))
        outhand4.write("{}\n".format(bs.dq[-1] if bs.new_trade_down_buy_index > -1 else None))
        outhand5.write("{}\n".format(bs.dq[-1] if bs.new_trade_down_sell_index > -1 else None))
        outhand6.write("{}\n".format(bs.wealth_new / bs.capital_init))
        outhand7.write("{}\n".format(bs.capital_new / bs.wealth_new))
        # 2. 策略只根据索引的状态 关闭
        bs.update_check_reset()
    outhand1.close()
    outhand2.close()
    outhand3.close()
    outhand4.close()
    outhand5.close()
    outhand6.close()
    outhand7.close()
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
    print("time:{}min".format((time.time() - stt) / 60))
    return final_ratio, len(datas), pow(final_ratio, len(datas))


def plot_datasig_package(datas):
    # 查看临时输出
    # pddata = pd.DataFrame({"Open": datas, "High": datas, "Low": datas, "Close": datas, "volumes": [1] * len(datas)})
    pddata = pd.DataFrame({"Open": datas, "High": datas, "Low": datas, "Close": datas})
    pddata["Open"] = np.nan
    pddata["Close"] = np.nan
    pddata["High"] = np.nan
    pddata["Low"] = np.nan
    # pddata["Open"] = pddata["Open"] * 0.95
    # pddata["Close"] = pddata["Close"] * 1
    # pddata["High"] = pddata["High"] * 1.05
    # pddata["Low"] = pddata["Low"] * 0.9
    orderl_pd = pd.date_range(start="19700101", periods=len(datas), freq='1D')
    pddata.set_index(orderl_pd, inplace=True)
    tmpfile1 = os.path.join(os.path.expanduser('~'), "tmp1.txt")
    with codecs.open(tmpfile1, "r", "utf8") as inhand:
        incont1 = inhand.readlines()
    tmpfile2 = os.path.join(os.path.expanduser('~'), "tmp2.txt")
    with codecs.open(tmpfile2, "r", "utf8") as inhand:
        incont2 = inhand.readlines()
    tmpfile3 = os.path.join(os.path.expanduser('~'), "tmp3.txt")
    with codecs.open(tmpfile3, "r", "utf8") as inhand:
        incont3 = inhand.readlines()
    tmpfile4 = os.path.join(os.path.expanduser('~'), "tmp4.txt")
    with codecs.open(tmpfile4, "r", "utf8") as inhand:
        incont4 = inhand.readlines()
    tmpfile5 = os.path.join(os.path.expanduser('~'), "tmp5.txt")
    with codecs.open(tmpfile5, "r", "utf8") as inhand:
        incont5 = inhand.readlines()
    tmpfile6 = os.path.join(os.path.expanduser('~'), "tmp6.txt")
    with codecs.open(tmpfile6, "r", "utf8") as inhand:
        incont6 = inhand.readlines()
    tmpfile7 = os.path.join(os.path.expanduser('~'), "tmp7.txt")
    with codecs.open(tmpfile7, "r", "utf8") as inhand:
        incont7 = inhand.readlines()
    pddata["sig"] = [np.nan if i1.strip() == "None" else datas[id1] for id1, i1 in enumerate(incont1)]
    pddata["ub"] = [
        np.nan if np.isnan(pddata["sig"][id1]) or i1.strip() == "None" or float(i1.strip()) < 0 else datas[id1]
        for id1, i1 in enumerate(incont2)]
    pddata["us"] = [
        np.nan if np.isnan(pddata["sig"][id1]) or i1.strip() == "None" or float(i1.strip()) < 0 else datas[id1]
        for id1, i1 in enumerate(incont3)]
    pddata["db"] = [
        np.nan if np.isnan(pddata["sig"][id1]) or i1.strip() == "None" or float(i1.strip()) < 0 else datas[id1]
        for id1, i1 in enumerate(incont4)]
    pddata["ds"] = [
        np.nan if np.isnan(pddata["sig"][id1]) or i1.strip() == "None" or float(i1.strip()) < 0 else datas[id1]
        for id1, i1 in enumerate(incont5)]
    wealth_plot = [float(i1.strip()) for i1 in incont6]
    ratio_plot = [float(i1.strip()) for i1 in incont7]
    # pddata["ub"] = [
    #     np.nan if np.isnan(pddata["sig"][id1]) or i1.strip() == "None" or float(i1.strip()) < 0 else pddata.iloc[id1, 0]
    #     for id1, i1 in enumerate(incont2)]
    # pddata["us"] = [
    #     np.nan if np.isnan(pddata["sig"][id1]) or i1.strip() == "None" or float(i1.strip()) < 0 else pddata.iloc[id1, 1]
    #     for id1, i1 in enumerate(incont3)]
    # pddata["db"] = [
    #     np.nan if np.isnan(pddata["sig"][id1]) or i1.strip() == "None" or float(i1.strip()) < 0 else pddata.iloc[id1, 3]
    #     for id1, i1 in enumerate(incont4)]
    # pddata["ds"] = [
    #     np.nan if np.isnan(pddata["sig"][id1]) or i1.strip() == "None" or float(i1.strip()) < 0 else pddata.iloc[id1, 2]
    #     for id1, i1 in enumerate(incont5)]
    pddata["sig"][0] = 1
    pddata["ub"][0] = 1
    pddata["us"][0] = 1
    pddata["db"][0] = 1
    pddata["ds"][0] = 1
    # print(pddata), ratio_plot
    plot_stock_sig(pddata, [datas, wealth_plot], pddata[["sig", "ub", "us", "db", "ds"]])


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
    n, m, beta = 1000, 1, 1.8
    # datas = generate_curve(n, m, beta, scale=0.01, plotsig=True)
    # datas = generate_curve(n, m, beta, scale=0.01, plotsig=False)
    datas = generate_simucurve(n, plotsig=True)
    # datas = generate_simucurve(n, plotsig=False)
    print(datas, datas[-1])
    exit()
    final_ratio = best_turtle_gene(datas, win=10, up_sell=[0.05], down_sell=[-0.01], up_buy=[0.01, 0.02],
                                   down_buy=[-0.05],
                                   plotsig=False)
    print(final_ratio)
    # # 查看临时输出
    plot_datasig_package(datas)
    # plot_curve(list(range(len(incont1))), [incont1, incont2, incont3, incont4, incont5],
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
