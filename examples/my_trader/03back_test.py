from vnpy.app.cta_strategy.backtesting import BacktestingEngine, OptimizationSetting
from vnpy.app.cta_strategy.base import BacktestingMode
from datetime import datetime
from atr_rsi_strategy import AtrRsiStrategy

engine = BacktestingEngine()
engine.set_parameters(
    vt_symbol="RU88.SHFE",
    # 必须设
    interval="1m",
    start=datetime(2019, 1, 1),
    end=datetime(2019, 4, 1),
    rate=0.5 / 10000,
    slippage=5,
    size=10,
    pricetick=5,
    capital=1_000_000,
    # 设置tick模式
    mode=BacktestingMode.TICK
)
engine.add_strategy(AtrRsiStrategy, {})

# 回测
engine.load_data()
engine.run_backtesting()
df = engine.calculate_result()
engine.calculate_statistics()
engine.show_chart()

# 打印每笔信息
trades = engine.trades
for value in trades.values():
    print("时间:",value.datetime,value.direction.value,value.offset.value, "价格：",value.price, "数量：",value.volume)
    if value.offset.value == "平":
        print("---------------------------------------------------------")