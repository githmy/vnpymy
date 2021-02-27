import numpy as np
import pandas as pd
import scipy.special as sc_special
from surf.basic_mlp import plot_curve
import matplotlib.pyplot as plt
import time


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
    if plotsig:
        tlen = len(ys)
        x = list(range(tlen))
        titles = "levy curve"
        # 检查画图
        plot_curve(x, [ys], titles)
    return ys


def strategy_keep50(datas, keep_cap=0.5, plotsig=False):
    """根据曲线生成策略参数: 持仓50%"""

    capital_old = 1.0
    stock_mount_old = 0.0
    price = 0.0
    wealth_old = capital_old + price * stock_mount_old
    wealths = []
    for price in datas:
        # 根据昨天的总额，按今天的价格调配
        wealth_old = capital_old + price * stock_mount_old
        capital_now = wealth_old * keep_cap
        stock_mount_now = (wealth_old - capital_now) / price
        capital_old = capital_now
        stock_mount_old = stock_mount_now
        wealths.append(wealth_old)
    if plotsig:
        tlen = len(datas)
        x = list(range(tlen))
        titles = ["stock", "wealth"]
        # 检查画图
        plot_curve(x, [datas, wealths], titles)
    return wealth_old


def main():
    """
    根据历史信息，超参数
    :return: 判断 板块 标的
    """
    # 1. 莱维曲线 参数提取，周期曲线复制
    np.random.seed(113)
    # TODO: 1. 不同的持仓位 2. 不同的幅值比例
    # n, m, beta = 100000000, 1, 1.8
    n, m, beta = 10, 1, 1.8
    datas = generate_curve(n, m, beta, scale=0.01, plotsig=False)
    final_profit = []
    for i1 in range(1, 100):
        stt = time.time()
        keep_cap = 0.01 * i1
        print(f"keep cap {keep_cap}")
        wealths = strategy_keep50(datas, keep_cap=keep_cap, plotsig=False)
        final_profit.append([wealths, datas[-1], wealths / datas[-1]])
        print((time.time()-stt)/60)
    pdobj = pd.DataFrame(final_profit)
    print(pdobj)
    pdobj[2]=np.log10(pdobj[2])
    print(pdobj)
    pdobj[2].plot()
    plt.show()
    # 1. 历史回测，压力模拟。
    # 2. 莱维曲线 和资金净流量的方向，历史价位和仓位控制，100个模拟统计，查看分布
    # 2. 资金净流量模型，m2延迟，GDP，国际利率差
    # 3. 资金流入，筹码集中的 交易市场，板块，个股
    # 3. 量比，委托比，内外盘 历史记录
    # 5. m0: 流通中现金，即银行体系以外的流通现金
    # 5. m1: 狭义货币供应量，即m0 + 企事业单位活期存款。
    # 5. m2: 广义货币供应量，即m1 + 企事业单位定期存款 + 居民储蓄存款。
    pass


if __name__ == '__main__':
    main()
