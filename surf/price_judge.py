import numpy as np
import math
import pandas as pd
from scipy.stats import norm
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, subplot, show, stem, title

"""
data_rules：数据分析规律
   bar_static_profit: log收益统计，分布
   bar_static_profit_is_normal: log收益是否正态分布
   bar_static_profit_mu_std_evolution: 均值方差变动演化
   
option_pricing_discrete：离散定价公式

black_scholes_option：期权定价公式

"""


class data_rules():
    def __init__(self):
        pass

    def bar_static_profit(self):
        """
        历史收益率
        均值 u = 1/n *sigma( \bar x)
        标准差 sigma^2 = 1/n * sum(xi - u)^2
        ln(X_{s+1}/X_{s}) -- N(u, sigma^2)  是否为常数，随时间不变化
        """
        data = pd.read_csv("../data/TSLA.csv")  # DataFrame
        close = data["Close"]
        # print(close)
        ret = [math.log(close[i + 1] / close[i]) for i in range(len(close) - 1)]
        # print(ret)
        # 1. 曲线
        plt.figure(0)
        plt.plot(close)
        # 2. log 收益率
        plt.figure(1)
        plt.plot(ret)
        # 3. 统计收益率分布
        plt.figure(2)
        plt.hist(ret, 100)
        plt.show()

    def bar_static_profit_is_normal(self):
        """收益率 均值标准差 是否正太分布？ 答不是"""
        data = pd.read_csv("../data/TSLA.csv")  # DataFrame
        days = 252
        close = data["Close"]
        # print(close)
        close = np.array(close)
        n = len(close)
        ret = np.log(close[1:n] / close[0:n - 1])
        n = len(ret)
        # print(ret)
        mu = np.mean(ret)
        sigma2 = np.sum((ret - mu) ** 2) / (n - 1)
        sigma = math.sqrt(sigma2)
        sigma_pd = pd.Series.std(pd.Series(ret))
        print(sigma, sigma_pd)
        # sigma_np = np.std(ret)
        # print(sigma, sigma_pd, sigma_np)
        plt.figure(0)
        x = np.linspace(-0.2, 0.2, 101)
        y = stats.norm.pdf(x, mu, sigma)
        plt.hist(ret, 100, density=True)
        plt.plot(x, y)
        print('mu:', mu)
        print('sigma:', sigma)
        print('---------------------------------------------------------------------------------------------')
        print('mu*days:', mu * days)
        print('sigma*sqrt(days):', sigma * np.sqrt(days))
        m = 252
        num = int(n / m)
        ret = ret[-num * m:-1]
        MU = np.zeros(num)
        SIGMA = np.zeros(num)
        plt.figure(1)
        for i in range(num):
            data0 = ret[i * m:(i + 1) * m]
            MU[i] = np.mean(data0)
            SIGMA[i] = pd.Series.std(pd.Series(data0))
            y = stats.norm.pdf(x, MU[i], SIGMA[i])
            plt.subplot(2, 5, i + 1)
            plt.hist(data0, 30, density=True)
            plt.plot(x, y)
        print('---------------------------------------------------------------------------------------------')
        print('MU:', MU)
        print('SIGMA:', SIGMA)
        MU *= days
        SIGMA *= np.sqrt(days)
        print('---------------------------------------------------------------------------------------------')
        print('MU*days:', MU)
        print('SIGMA*sqrt(days):', SIGMA)
        plt.subplot(3, 1, 1)
        plt.plot(ret)
        plt.subplot(3, 1, 2)
        plt.plot(MU)
        plt.subplot(3, 1, 3)
        plt.plot(SIGMA)
        plt.show()

    def bar_static_profit_mu_std_evolution(self):
        """收益率 均值标准差 演变，是否稳定？ 答不稳定"""

        # 计算对数收益率
        def logreturn(x):
            n = len(x)
            ret = np.log(x[1:n] / x[0:n - 1])
            return ret

        # 估计参数 无偏估计
        def estimate(ret):
            mu = np.mean(ret)
            sigma_pd = pd.Series.std(pd.Series(ret))
            return mu, sigma_pd

        # 分段估计参数
        def estimate_intervals(ret, m):
            n = len(ret)
            num = int(n / m)

            ret = ret[-num * m:]
            MU = np.zeros(num)
            SIGMA = np.zeros(num)
            for i in range(num):
                data0 = ret[i * m:(i + 1) * m]
                MU[i] = np.mean(data0)
                SIGMA[i] = pd.Series.std(pd.Series(data0))

            return MU, SIGMA

        # 分析数据
        def analyse(file, m=30):
            data = pd.read_csv(file)  # DataFrame
            days = 252

            close = data["Close"]

            close = np.array(close)
            ret = logreturn(close)

            mu, sigma = estimate(ret)

            print('mu:', mu)
            print('sigma:', sigma)

            print('---------------------------------------------------------------------------------------------')
            print('mu*days:', mu * days)
            print('sigma*sqrt(days):', sigma * np.sqrt(days))

            MU, SIGMA = estimate_intervals(ret, m)

            print('---------------------------------------------------------------------------------------------')
            print('MU:', MU)
            print('SIGMA:', SIGMA)

            MU *= days
            SIGMA *= np.sqrt(days)

            print('---------------------------------------------------------------------------------------------')
            print('MU*days:', MU)
            print('SIGMA*sqrt(days):', SIGMA)

            # 画图
            subplot(4, 1, 1)
            plot(close)
            title('close data')

            subplot(4, 1, 2)
            plot(ret)
            title('log return')

            subplot(4, 1, 3)
            stem(MU)
            title('MU')

            subplot(4, 1, 4)
            stem(SIGMA)
            title('SIGMA')
            show()

        m = 252
        analyse("../data/TSLA.csv", m)
        analyse("../data/IBM.csv", m)
        analyse("../data/HAL.csv", m)


def option_pricing_discrete(price_now, price_force, rise_up, rise_dn, rate, t):
    """
    根据当前price_now，执行价格，涨幅, 跌幅，利率，时间差 定出期权价格
    a = rise_up - 1 
    b = 1 - rise_dn
    proba_up / proba_dn = a / b
    price_expection_future = price_now * math.e(rate * t)
    price_expection_future = (proba_up * rise_up + proba_dn * rise_dn) * price_now
    variance_future = (proba_up * rise_up ** 2 + proba_dn * rise_dn ** 2) * price_now ** 2 - price_expection_future ** 2
    
    proba_up = (math.e(rate * t)-rise_dn)/(rise_up-rise_dn)  
    """
    proba_up = (math.e(rate * t) - rise_dn) / (rise_up - rise_dn)
    proba_dn = 1 - proba_up
    price_up = price_now * rise_up
    price_dn = price_now * rise_dn
    # 截止日之前，跌多了可以高卖
    fall_american = False
    # 只有截止日，跌多了可以高卖
    fall_europe = False
    # 截止日之前，涨高了可以低买
    rise_american = False
    # 只有截止日，涨高了可以低买
    rise_europe = True
    if rise_europe:
        profit_high = price_up - price_force if price_up - price_force > 0 else 0
        profit_low = price_dn - price_force if price_dn - price_force > 0 else 0
        option4rise_europe = (proba_up * profit_high + proba_dn * profit_low) * math.e(rate * t)
        return option4rise_europe
    elif rise_american:
        option4rise_american = None
        return option4rise_american
    elif fall_american:
        profit_high = price_force - price_up if price_force - price_up > 0 else 0
        profit_low = price_force - price_dn if price_force - price_dn > 0 else 0
        option4fall_american = (proba_up * profit_high + proba_dn * profit_low) * math.e(rate * t)
        return option4fall_american
    elif fall_europe:
        option4fall_europe = None
        return option4fall_europe


def black_scholes_option(S, K, T, r, q, sigma, option='call'):
    """https://zhuanlan.zhihu.com/p/147007821
    S: spot price
    K: strike price
    T: time to maturity
    r: risk-free interest rate
    q: rate of continuous dividend
    sigma: standard deviation of price of underlying asset  
    """
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option == 'call':
        p = (S * np.exp(-q * T) * norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * norm.cdf(d2, 0.0, 1.0))
    elif option == 'put':
        p = (K * np.exp(-r * T) * norm.cdf(-d2, 0.0, 1.0) - S * np.exp(-q * T) * norm.cdf(-d1, 0.0, 1.0))
    else:
        return None
    return p


def main():
    pass


if __name__ == '__main__':
    main()
