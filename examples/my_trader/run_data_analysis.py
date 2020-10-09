#!/usr/bin/env python
# coding: utf-8


from vnpy.trader.constant import Exchange,Interval
from examples.data_analysis.data_analysis import DataAnalysis
from datetime import datetime
import matplotlib.pyplot as plt


herramiento = DataAnalysis()
herramiento.load_history(    
    symbol="603999",
    exchange=Exchange.SSE,
    interval=Interval.DAILY,
    # symbol="XBTUSD",
    # exchange=Exchange.BITMEX,
    # interval=Interval.MINUTE,
    start=datetime(2014, 9, 1),
    end=datetime(2018, 10, 30),
    rate = 8/10000,
    index_3to1 = ["ATR","CCI"],
    index_1to1 = ["STDDEV","SMA"],
    index_2to2 = ["AROON"],
    index_2to1 = ["AROONOSC"],
    index_4to1 = ["BOP"],
    window_index=30,
)

data = herramiento.base_analysis()

# 手绘交易图
herramiento.show_chart(data[:1500], boll_wide=2.8)

# 多时间周期分析
# intervals = ["5min","15min","30min","1h","2h","4h"]
intervals = ["1d"]
herramiento.multi_time_frame_analysis(intervals=intervals)

