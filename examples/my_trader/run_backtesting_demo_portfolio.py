#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
from importlib import reload

import vnpy.app.portfolio_strategy
reload(vnpy.app.portfolio_strategy)

from vnpy.app.portfolio_strategy import BacktestingEngine
from vnpy.trader.constant import Interval

import vnpy.app.portfolio_strategy.strategies.pair_trading_strategy as stg
reload(stg)
from vnpy.app.portfolio_strategy.strategies.pair_trading_strategy import PairTradingStrategy


engine = BacktestingEngine()
engine.set_parameters(
    vt_symbols=["y888.DCE", "p888.DCE"],
    interval=Interval.MINUTE,
    start=datetime(2019, 1, 1),
    end=datetime(2020, 4, 30),
    rates={
        "y888.DCE": 0/10000,
        "p888.DCE": 0/10000
    },
    slippages={
        "y888.DCE": 0,
        "p888.DCE": 0
    },
    sizes={
        "y888.DCE": 10,
        "p888.DCE": 10
    },
    priceticks={
        "y888.DCE": 1,
        "p888.DCE": 1
    },
    capital=1_000_000,
)

setting = {
    "boll_window": 20,
    "boll_dev": 1,
}
engine.add_strategy(PairTradingStrategy, setting)


engine.load_data()
engine.run_backtesting()
df = engine.calculate_result()
engine.calculate_statistics()
engine.show_chart()



