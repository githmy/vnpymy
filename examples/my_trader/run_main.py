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

def generate_targets():
    pass

def strategy4targets():
    pass

def main():
    # 1. 遍历所有分析方法，生成标的组。
    generate_targets()
    # 2. 遍历所有策略
    # 3. 遍历所有标的
    strategy4targets()
    pass

if __name__ == '__main__':
    main()