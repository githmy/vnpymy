from vnpy.trader.constant import (Exchange, Interval)
import pandas as pd
from vnpy.trader.database import database_manager
from vnpy.trader.object import (BarData, TickData)
from datetime import datetime
import pandas as pd
import os


def run_load_txt(loadpath):
    """
    遍历同一文件夹内所有txt文件，并且载入到数据库中
    """
    allfiles = os.listdir(loadpath)
    lenth = len(allfiles)
    for idn, file in enumerate(allfiles):
        if not file.endswith(".txt"):
            continue
        print(idn, lenth)
        txt_load(loadpath, file)


def txt_load(loadpath, file):
    """
    读取csv文件内容，并写入到数据库中    
    """
    print("载入文件：", file)
    fullfile = os.path.join(loadpath, file)
    exchange, symbol = file.replace(".", "#").split("#")[:2]
    if exchange == "SH":
        exchange = Exchange.SSE
    elif exchange == "SZ":
        exchange = Exchange.SZSE
        raise Exception("未匹配的交易所类型")
    else:
        raise Exception("未匹配的交易所类型")
    with open(fullfile, "r") as f:
        filelines = f.readlines()
        if len(filelines) < 4:
            return None
        res = filelines[:2]
        if res[0].split()[2] == "日线":
            interval = Interval.DAILY
        else:
            raise Exception("未匹配的间隔类型")
        tlist = [[col.strip() for col in item.split("\t")] for item in filelines[2:-1]]
        df = pd.DataFrame(tlist)
        df.columns = res[1].split()
        data = []
        if df is not None:
            for ix, row in df.iterrows():
                date = datetime.strptime(row["日期"], '%Y/%m/%d')
                bar = BarData(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    datetime=date,
                    open_price=row["开盘"],
                    high_price=row["最高"],
                    low_price=row["最低"],
                    close_price=row["收盘"],
                    volume=row["成交量"],
                    gateway_name="XTP"
                )
                data.append(bar)
        if data:
            database_manager.save_bar_data(data)
            print("单标的 插入数据 数量：", df.shape[0])


if __name__ == "__main__":
    loadpath = os.path.join("E:\\", "project", "my", "xtpjst", "T0002", "export")
    run_load_txt(loadpath)
