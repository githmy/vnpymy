from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import talib

from vnpy.trader.constant import Exchange, Interval
from vnpy.trader.database import database_manager

warnings.filterwarnings("ignore")


class DataAnalysis:
    def __init__(self):
        """"""
        self.symbol = ""
        self.exchange = None
        self.interval = None
        self.start = None
        self.end = None
        self.rate = 0.0

        self.window_volatility = 20
        self.window_index = 20

        self.orignal = pd.DataFrame()

        self.index_1to1 = []
        self.index_2to2 = []
        self.index_3to1 = []
        self.index_2to1 = []
        self.index_4to1 = []
        self.intervals = []

        self.results = {}

    def load_history(
            self,
            symbol: str,
            exchange: Exchange,
            interval: Interval,
            start: datetime,
            end: datetime,
            rate: float = 0.0,
            index_1to1: list = None,
            index_2to2: list = None,
            index_3to1: list = None,
            index_2to1: list = None,
            index_4to1: list = None,
            window_index: int = 20,
            window_volatility: int = 20,

    ):
        """"""
        output("开始加载历史数据")

        self.window_volatility = window_volatility
        self.window_index = window_index
        self.rate = rate
        self.index_1to1 = index_1to1
        self.index_2to2 = index_2to2
        self.index_3to1 = index_3to1
        self.index_2to1 = index_2to1
        self.index_4to1 = index_4to1

        # Load history data from database
        bars = database_manager.load_bar_data(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start=start,
            end=end,

        )
        output(f"历史数据加载完成，数据量：{len(bars)}")

        # Generate history data in DataFrame
        t = []
        o = []
        h = []
        l = []  # noqa
        c = []
        v = []

        for bar in bars:
            time = bar.datetime
            open_price = bar.open_price
            high_price = bar.high_price
            low_price = bar.low_price
            close_price = bar.close_price
            volume = bar.volume

            t.append(time)
            o.append(open_price)
            h.append(high_price)
            l.append(low_price)
            c.append(close_price)
            v.append(volume)

        self.orignal["open"] = o
        self.orignal["high"] = h
        self.orignal["low"] = l
        self.orignal["close"] = c
        self.orignal["volume"] = v
        self.orignal.index = t

    def base_analysis(self, df: DataFrame = None):
        """"""
        if df is None:
            df = self.orignal

        if df is None:
            output("数据为空，请输入数据")

        close_price = df["close"]

        output("第一步:画出行情图，检查数据断点")
        close_price.plot(figsize=(20, 8), title="close_price")
        plt.show()

        # 是否自相关
        random_test(close_price)
        # 时间序列 平稳性
        stability_test(close_price)
        # 绘制 自相关
        autocorrelation_test(close_price)
        # 相对波动率
        self.relative_volatility_analysis(df)
        # 涨幅
        self.growth_analysis(df)
        # 计算相关技术指标
        self.calculate_index(df)

        return df

    def relative_volatility_analysis(self, df: DataFrame = None):
        """
        相对波动率
        """
        output("第五步：相对波动率分析")
        df["volatility"] = talib.ATR(
            np.array(df["high"]),
            np.array(df["low"]),
            np.array(df["close"]),
            self.window_volatility
        )

        df["fixed_cost"] = df["close"] * self.rate
        df["relative_vol"] = df["volatility"] - df["fixed_cost"]

        df["relative_vol"].plot(figsize=(20, 6), title="relative volatility")
        plt.show()

        df["relative_vol"].hist(bins=200, figsize=(20, 6), grid=False)
        plt.show()

        statitstic_info(df["relative_vol"])

    def growth_analysis(self, df: DataFrame = None):
        """
        百分比K线变化率
        """
        output("第六步：涨幅 变化率分析")
        df["pre_close"] = df["close"].shift(1).fillna(0)
        df["g%"] = 100 * (df["close"] - df["pre_close"]) / df["pre_close"]

        df["g%"].plot(figsize=(20, 6), title="growth", ylim=(-5, 5))
        plt.show()

        df["g%"].hist(bins=200, figsize=(20, 6), grid=False)
        plt.show()

        statitstic_info(df["g%"])

    def calculate_index(self, df: DataFrame = None):
        """"""
        output("第七步：计算相关技术指标，返回DataFrame\n")

        if self.index_1to1:
            for i in self.index_1to1:
                func = getattr(talib, i)
                df[i] = func(
                    np.array(df["close"]),
                    self.window_index
                )

        if self.index_3to1:
            for i in self.index_3to1:
                func = getattr(talib, i)
                df[i] = func(
                    np.array(df["high"]),
                    np.array(df["low"]),
                    np.array(df["close"]),
                    self.window_index
                )

        if self.index_2to2:
            for i in self.index_2to2:
                func = getattr(talib, i)
                result_down, result_up = func(
                    np.array(df["high"]),
                    np.array(df["low"]),
                    self.window_index
                )
                up = i + "_UP"
                down = i + "_DOWN"
                df[up] = result_up
                df[down] = result_down

        if self.index_2to1:
            for i in self.index_2to1:
                func = getattr(talib, i)
                df[i] = func(
                    np.array(df["high"]),
                    np.array(df["low"]),
                    self.window_index
                )

        if self.index_4to1:
            for i in self.index_4to1:
                func = getattr(talib, i)
                df[i] = func(
                    np.array(df["open"]),
                    np.array(df["high"]),
                    np.array(df["low"]),
                    np.array(df["close"]),
                )

        return df

    def multi_time_frame_analysis(self, intervals: list = None, df: DataFrame = None):
        """"""
        if not intervals:
            output("请输入K线合成周期")
            return

        if df is None:
            df = self.orignal

        if df is None:
            output("请先加载数据")
            return

        for interval in intervals:
            output("------------------------------------------------")
            output(f"合成{interval}周期K先并开始数据分析")

            data = pd.DataFrame()
            data["open"] = df["open"].resample(interval, how="first")
            data["high"] = df["high"].resample(interval, how="max")
            data["low"] = df["low"].resample(interval, how="min")
            data["close"] = df["close"].resample(interval, how="last")
            data["volume"] = df["volume"].resample(interval, how="sum")

            result = self.base_analysis(data)
            self.results[interval] = result

    def show_chart(self, data, boll_wide):
        """手绘交易图"""
        data["boll_up"] = data["SMA"] + data["STDDEV"] * boll_wide
        data["boll_down"] = data["SMA"] - data["STDDEV"] * boll_wide

        up_signal = []
        down_signal = []
        len_data = len(data["close"])
        for i in range(1, len_data):
            if data.iloc[i]["close"] > data.iloc[i]["boll_up"] and data.iloc[i - 1]["close"] < data.iloc[i - 1][
                "boll_up"]:
                up_signal.append(i)

            elif data.iloc[i]["close"] < data.iloc[i]["boll_down"] and data.iloc[i - 1]["close"] > data.iloc[i - 1][
                "boll_down"]:
                down_signal.append(i)

        plt.figure(figsize=(20, 8))
        close = data["close"]
        plt.plot(close, lw=1)
        plt.plot(close, '^', markersize=5, color='r',
                 label='UP signal', markevery=up_signal)
        plt.plot(close, 'v', markersize=5, color='g',
                 label='DOWN signal', markevery=down_signal)
        plt.plot(data["boll_up"], lw=0.5, color="r")
        plt.plot(data["boll_down"], lw=0.5, color="g")
        plt.legend()
        plt.show()

        data["ATR"].plot(figsize=(20, 3), title="ATR")
        plt.show()


def random_test(close_price):
    """ 函数检验无自相关 """
    acorr_result = acorr_ljungbox(close_price, lags=1)
    p_value = acorr_result[1]
    if p_value < 0.05:
        output("第二步：随机性检验：非纯随机性")
    else:
        output("第二步：随机性检验：纯随机性")
    output(f"白噪声检验结果:{acorr_result}\n")


def stability_test(close_price):
    """时间序列不平稳 
    第一部分是τ（tao）统计量的值。
    第二部分是p_value的值。
    第三部分是结果使用的延迟阶数。
    第四部分是ADF回归和计算临界值所使用的观察次数。
    第五个是配合第一个一起看的，是在99%，95%，90%置信区间下的临界的ADF检验的值。
    第六部分是最大的信息准则的值（如果autolag 非空），也就是AIC或者BIC的值。
    当我们看序列是否平稳的结果时，一般首先看第二部分的p_value值。如果p_value值比0.05小，证明有单位根，也就是说序列平稳。如果p_value比0.05大则证明非平稳。
    源码里有一句note，如果p_value接近于0.05时，则要通过临界值进行判断。也就是说如果p_value接近于0.05就要通过第一部分τ（tao）统计量的值和第五部分的临界值进行对比。τ（tao）统计量的值比临界值小，就证明平稳，反之就是非平稳。我的检测结果τ（tao）统计量的值再临界值5%-10%之间。比5%的临界值大。这里的1%，5%，10%对应的是99%，95%，90%置信区间。
    ADF检验只适合AR(P)模型。
    且对方差齐性效果好，对异方差性效果不佳。异方差可用PP检验。
    如果第二个值比第五个值小证明平稳，反正证明不平稳。根据结果看出来，你的数据不平稳。
    """
    statitstic = ADF(close_price)
    t_s = statitstic[1]
    t_c = statitstic[4]["10%"]

    if t_s > t_c:
        output("第三步：平稳性检验：存在单位根，时间序列不平稳")
    else:
        output("第三步：平稳性检验：不存在单位根，时间序列平稳")

    output(f"ADF检验结果：{statitstic}\n")


def autocorrelation_test(close_price):
    """"""
    output("第四步：画出自相关性图，观察自相关特性")

    plot_acf(close_price, lags=60)
    plt.show()

    plot_pacf(close_price, lags=60).show()
    plt.show()


def statitstic_info(df):
    """"""
    mean = round(df.mean(), 4)
    median = round(df.median(), 4)
    output(f"样本平均数：{mean}, 中位数: {median}")

    skew = round(df.skew(), 4)
    kurt = round(df.kurt(), 4)

    if skew == 0:
        skew_attribute = "对称分布"
    elif skew > 0:
        skew_attribute = "分布偏左"
    else:
        skew_attribute = "分布偏右"

    if kurt == 0:
        kurt_attribute = "正态分布"
    elif kurt > 0:
        kurt_attribute = "分布陡峭"
    else:
        kurt_attribute = "分布平缓"

    output(f"偏度为：{skew}，属于{skew_attribute}；峰度为：{kurt}，属于{kurt_attribute}\n")


def output(msg):
    """
    Output message of backtesting engine.
    """
    print(f"{datetime.now()}\t{msg}")
