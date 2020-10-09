"""
Average Drawdown, 平均回撤（ADD）
Linearly Weighted Drawdown, 线性加权回撤（lwDD）
Average Squared Drawdown, 均方回撤（ADD^2)
Trend Weighted Drawdown, 趋势加权回撤（twDD)
Maximum Drawdown, 最大回撤（MDD）
End-of-period Drawdown, 期末回撤（eopDD）
"""
from datetime import datetime
import cProfile

from vnpy.app.cta_strategy.backtesting import BacktestingEngine, OptimizationSetting
from vnpy.app.cta_strategy.strategies.atr_rsi_strategy import AtrRsiStrategy


def runBacktesting():
    engine = BacktestingEngine()
    engine.set_parameters(
        vt_symbol="IF888.CFFEX",
        interval="1m",
        start=datetime(2010, 1, 1),
        end=datetime(2019, 12, 30),
        rate=0.3 / 10000,
        slippage=0.2,
        size=300,
        pricetick=0.2,
        capital=1_000_000,
    )
    engine.add_strategy(AtrRsiStrategy, {})
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    engine.calculate_statistics()
    engine.show_chart()


if __name__ == '__main__':
    runBacktesting()
    # 分析结果输出
    # python -m cProfile -o result.out run.py
    # 结果 转 图片
    # python -m gprof2dot -f pstats result.out | dot -Tpng -o result.png
